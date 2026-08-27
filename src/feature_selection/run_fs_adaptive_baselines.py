"""Baseline transfer-thích-nghi để CÔ LẬP đóng góp relevance của RG-SCSO (rev #1/Q4).

Chạy PSO/GWO nhị phân trang bị transfer function V-shaped ĐÃ CÔNG BỐ nhưng KHÔNG
có relevance per-feature — dưới CÙNG giao thức wrapper với RG-SCSO (KNN k=5, 5-fold
CV, fitness = 0.99·err + 0.01·tỉ_lệ_feature, biên [-1,1], pop=30, iter=500, ngân
sách NFE = pop×iter, seed = RANDOM_SEED_BASE + run_id). Nếu RG-SCSO vẫn vượt các
baseline này -> lợi thế đến từ relevance per-feature, KHÔNG phải từ binarize
V-shaped/thích nghi chung chung.

BỐN CẤU HÌNH (2 optimizer × 2 họ transfer):
    bPSO-TVT, bGWO-TVT   : Islam time-varying |tanh(τ·x)|, τ 4 -> 0.01
    bPSO-V4 , bGWO-V4    : Teng V4 |(2/π)arctan((π/2)x)| cố định

Giao thức KHÓA TRƯỚC (spec pre-registration): 18 dataset × 30 run, khớp bảng chính.
KHÔNG tinh chỉnh tham số cho số đẹp (spec 8.1/4.2); báo cáo thắng/thua trung thực.

Output: experiments/results_fs_adaptive_baselines/fs_adaptive_baselines_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_adaptive_baselines [--smoke]
        [--datasets Zoo,Sonar,...] [--configs bPSO-TVT,bGWO-TVT,...] [--runs N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.binary_baselines import (
    BinaryGWO,
    BinaryPSO,
    islam_tvt_pflip,
    teng_v4_pflip,
)
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_adaptive_baselines")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_adaptive_baselines_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0

# 18 dataset KHỚP bảng chính fs_results.csv (so sánh cùng protocol/seed).
ALL_DATASETS = [
    "BreastEW", "ColonCancer", "Diabetes", "GermanCredit", "HeartDisease",
    "IonosphereEW", "KrVsKpEW", "Leukemia", "Lymphography", "M-of-n",
    "Parkinsons", "Sonar", "SpectEW", "TicTacToe", "Vote", "WDBC",
    "WaveformEW", "Zoo",
]

# config -> (lớp optimizer, hàm transfer)
CONFIGS = {
    "bPSO-TVT": (BinaryPSO, islam_tvt_pflip),
    "bGWO-TVT": (BinaryGWO, islam_tvt_pflip),
    "bPSO-V4": (BinaryPSO, teng_v4_pflip),
    "bGWO-V4": (BinaryGWO, teng_v4_pflip),
}


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    """(config, dataset, run_id) đã hoàn tất -> để resume, không chạy lại."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["config"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    config, ds, run_id = task["config"], task["dataset"], task["run_id"]
    optimizer_cls, transfer = CONFIGS[config]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    result = optimizer_cls(
        obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
        eval_mask=eval_mask, transfer=transfer,
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    return {
        "config": config, "dataset": ds, "run_id": run_id,
        "fitness": final["fitness"], "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "runtime_seconds": result["runtime"],
    }


def _append_rows(rows: list[dict]) -> None:
    """Ghi bổ sung (giữ resume an toàn nếu job bị ngắt giữa chừng)."""
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 run, Zoo, 1 config — kiểm tra wiring (KHÔNG phải kết quả).")
    ap.add_argument("--datasets", type=str, default=None,
                    help="Danh sách dataset ngăn cách bởi dấu phẩy (mặc định cả 18).")
    ap.add_argument("--configs", type=str, default=None,
                    help="Danh sách config (mặc định cả 4).")
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS,
                    help=f"Số run độc lập (mặc định {NUM_INDEPENDENT_RUNS}).")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, configs, n_runs, workers = ["Zoo"], ["bPSO-TVT"], 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS
        configs = args.configs.split(",") if args.configs else list(CONFIGS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"config": c, "dataset": ds, "run_id": r}
        for c in configs for ds in datasets for r in range(n_runs)
        if (c, ds, r) not in done
    ]
    total = len(configs) * len(datasets) * n_runs
    print(f"Adaptive baselines: {len(configs)} config x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="adaptive-bl"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['config']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
