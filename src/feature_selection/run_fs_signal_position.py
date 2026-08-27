"""Ablation cô lập VỊ TRÍ tiêm tín hiệu relevance (Q1 review Priority 2 / W7)
— lỗ hổng causal được đánh giá là LỚN NHẤT của bài: hiện tại bài chỉ so sánh
*relevance-guided binarization* vs. *relevance-agnostic binarization*, chưa
trả lời được câu hỏi cốt lõi mà bài tuyên bố: "đưa MI vào binarization
operator có thực sự quan trọng hơn đưa MI vào initialization hay objective
hay không?"

CHUỖI 5 BƯỚC (đúng theo đề xuất review, Priority 2):
    1. Random init,      no RMS   (= NoImprovement, đã có trong ablation chính)
    2. MI-guided init,   no RMS   (MỚI — tín hiệu ở INITIALIZATION)
    3. MI-weighted obj.,  no RMS  (MỚI — tín hiệu ở OBJECTIVE)
    4. MI-guided transfer (RMS), no UMR (= NoUMR, đã có trong ablation chính)
    5. RMS + UMR          (= Full, đã có trong ablation chính)

Nếu bước #2 hoặc #3 đạt độ nhỏ gọn tương đương bước #4/#5, novelty trung tâm
của bài ("đặt tín hiệu ở binarization interface quan trọng hơn init/objective")
sẽ suy yếu đáng kể — đây CHÍNH LÀ điều thí nghiệm này kiểm định trung thực,
không phải điều được giả định trước.

Giao thức khớp bảng ablation chính: KNN, 5-fold CV, seed = BASE + run_id, 30
run, trên 5 dataset đại diện (Leukemia, ColonCancer, Sonar, WDBC, Zoo).

Output: experiments/results_fs_signal_position/fs_signal_position_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_signal_position
        [--smoke] [--datasets ...] [--runs N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, NUM_INDEPENDENT_RUNS, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.rg_scso_signal_position import RGSCSOSignalPosition
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_signal_position")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_signal_position_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Leukemia", "ColonCancer", "Sonar", "WDBC", "Zoo"]

# step_name -> kwargs cho RGSCSOSignalPosition (thứ tự khớp đúng 5 bước review).
STEPS = {
    "1_RandomInit_NoRMS": dict(injection="transfer", use_rms=False, use_orl=False, use_umr=False),
    "2_MIInit_NoRMS": dict(injection="init", use_orl=False, use_umr=False),
    "3_MIObjective_NoRMS": dict(injection="objective", use_orl=False, use_umr=False),
    "4_MITransfer_RMS_NoUMR": dict(injection="transfer", use_rms=True, use_orl=False, use_umr=False),
    "5_RMS_UMR_Full": dict(injection="transfer", use_rms=True, use_orl=False, use_umr=True),
}


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["step"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    step, ds, run_id = task["step"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    result = RGSCSOSignalPosition(
        obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
        X=X, y=y, eval_mask=eval_mask, **STEPS[step],
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    return {
        "step": step, "dataset": ds, "run_id": run_id,
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
                     help="1 run, Zoo, tất cả 5 bước — kiểm tra wiring.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, steps, n_runs, workers = ["Zoo"], list(STEPS), 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        steps = list(STEPS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"step": s, "dataset": ds, "run_id": r}
        for s in steps for ds in datasets for r in range(n_runs)
        if (s, ds, r) not in done
    ]
    total = len(steps) * len(datasets) * n_runs
    print(f"Signal-position: {len(steps)} step x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="signal-position"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['step']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
