"""Can thiệp nhân quả lên relevance prior (Q1 review Priority 3) — kiểm định
liệu claim "causally linked to relevance guidance" có đứng vững hay không.

Ý TƯỞNG: size-fair top-MI enrichment (mục Mechanism trong bài chính) chỉ là
BẰNG CHỨNG TƯƠNG QUAN — RG-SCSO chọn nhiều feature MI cao hơn ngẫu nhiên,
nhưng không chứng minh MI THẬT SỰ là NGUYÊN NHÂN gây ra độ nhỏ gọn/accuracy
quan sát được. Để kiểm định nhân quả thật, can thiệp trực tiếp lên CHÍNH
ánh xạ relevance rồi xem hiệu năng có sụp đổ không:

    RG-SCSO(MI thật)      : ρ_j = I(X_j;y)/H(y)              — baseline
    RG-SCSO(MI xáo trộn)  : permute(ρ) giữa các feature       — phá vỡ ánh xạ
                             feature<->relevance, GIỮ NGUYÊN phân phối ρ
    RG-SCSO(MI đảo cực)   : 1 - ρ                              — feature liên
                             quan thật thành "nhiễu" theo RMS và ngược lại

Nếu MI thật > MI xáo trộn/đảo cực về accuracy VÀ độ nhỏ gọn một cách có ý
nghĩa thống kê (Wilcoxon+Holm, so target = MI thật), claim nhân quả mới có
cơ sở giữ trong bài; nếu KHÔNG khác biệt, "causal" phải hạ xuống "correlational".

Giao thức KHỚP bảng chính: KNN wrapper, 5-fold CV, fitness = 0.99·err +
0.01·tỉ_lệ, biên [-1,1], pop=30, iter=500, seed = BASE + run_id, 30 run, trên
5 dataset đại diện (khớp tập ablation/robustness để nhất quán và tiết kiệm
compute — mở rộng ra 18 dataset là bước tiếp theo nếu kết quả sơ bộ ủng hộ).

Output: experiments/results_fs_shuffle_mi/fs_shuffle_mi_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_shuffle_mi [--smoke]
        [--datasets ...] [--runs N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from config import (
    KNN_NEIGHBORS,
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.rg_scso import RGSCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_shuffle_mi")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_shuffle_mi_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0

# Khớp tập ablation/robustness: đa dạng chiều/lớp/mẫu, tiết kiệm compute.
DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]

# algorithm-config -> prior method truyền cho RGSCSO(prior_method=...)
ALGOS = {
    "RG-SCSO-MI": "mi",
    "RG-SCSO-ShuffledMI": "shuffled_mi",
    "RG-SCSO-InvertedMI": "inverted_mi",
}


def _make_clf(seed: int):
    return KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    algo, ds, run_id = task["algorithm"], task["dataset"], task["run_id"]
    prior_method = ALGOS[algo]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id

    def clf_factory():
        return _make_clf(seed)

    obj_func = make_fitness_function(X, y, seed=seed, clf_factory=clf_factory)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed, clf_factory=clf_factory)["fitness"]

    result = RGSCSO(
        obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
        pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
        X=X, y=y, eval_mask=eval_mask, prior_method=prior_method,
    ).optimize()

    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed, clf_factory=clf_factory)
    return {
        "algorithm": algo, "prior_method": prior_method, "dataset": ds, "run_id": run_id,
        "fitness": final["fitness"], "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "runtime_seconds": result["runtime"],
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, 1 algo — kiểm tra wiring (KHÔNG phải kết quả).")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, algos, n_runs, workers = ["Zoo"], ["RG-SCSO-ShuffledMI"], 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        algos = list(ALGOS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": a, "dataset": ds, "run_id": r}
        for a in algos for ds in datasets for r in range(n_runs)
        if (a, ds, r) not in done
    ]
    total = len(algos) * len(datasets) * n_runs
    print(f"Shuffle-MI: {len(algos)} algo x {len(datasets)} dataset x {n_runs} run "
          f"= {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shuffle-mi"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
