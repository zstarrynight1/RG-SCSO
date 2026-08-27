"""Baseline SCSO-FS CÙNG HỌ (rev #2) — so RG-SCSO trực tiếp với các biến thể
SCSO-based feature selection, đúng dòng thuật toán mà bài tuyên bố "gap".

Chạy bSCSO-S và bSCSO-OBL (xem src/algorithms/scso_fs_baselines.py) dưới CÙNG giao
thức wrapper với bảng chính: KNN k=5, 5-fold CV, fitness = 0.99·err + 0.01·tỉ_lệ,
biên [-1,1], pop=30, iter=500, ngân sách NFE = pop×iter, seed = BASE + run_id.
18 dataset × 30 run — KHÓA TRƯỚC, KHÔNG tinh chỉnh cho số đẹp (spec 8.1/4.2).

Kết quả ghép với fs_results.csv (cùng cột) để đưa vào Table II–VII như baseline
họ SCSO. ĐỌC caveat sourcing trong scso_fs_baselines.py trước khi công bố.

Output: experiments/results_fs_scso_family/fs_scso_family_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_scso_family [--smoke]
        [--datasets Zoo,Sonar,...] [--configs bSCSO-S,bSCSO-OBL] [--runs N]
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
from src.algorithms.scso_fs_baselines import BinarySCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_scso_family")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_scso_family_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0

# 18 dataset KHỚP bảng chính fs_results.csv (cùng protocol/seed).
ALL_DATASETS = [
    "BreastEW", "ColonCancer", "Diabetes", "GermanCredit", "HeartDisease",
    "IonosphereEW", "KrVsKpEW", "Leukemia", "Lymphography", "M-of-n",
    "Parkinsons", "Sonar", "SpectEW", "TicTacToe", "Vote", "WDBC",
    "WaveformEW", "Zoo",
]

# config -> kwargs cho BinarySCSO (khác nhau ở transfer + OBL, cùng SCSO move).
CONFIGS = {
    "bSCSO-S": {"transfer_kind": "s", "use_obl": False},
    "bSCSO-OBL": {"transfer_kind": "v", "use_obl": True},
}


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    config, ds, run_id = task["algorithm"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    result = BinarySCSO(
        obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
        eval_mask=eval_mask, **CONFIGS[config],
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    return {
        "algorithm": config, "dataset": ds, "run_id": run_id,
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
                    help="1 run, Zoo, 1 config — kiểm tra wiring (KHÔNG phải kết quả).")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--configs", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, configs, n_runs, workers = ["Zoo"], ["bSCSO-OBL"], 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS
        configs = args.configs.split(",") if args.configs else list(CONFIGS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": c, "dataset": ds, "run_id": r}
        for c in configs for ds in datasets for r in range(n_runs)
        if (c, ds, r) not in done
    ]
    total = len(configs) * len(datasets) * n_runs
    print(f"SCSO-family baselines: {len(configs)} config x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="scso-family"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
