
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

                                                                         
ALL_DATASETS = [
    "BreastEW", "ColonCancer", "Diabetes", "GermanCredit", "HeartDisease",
    "IonosphereEW", "KrVsKpEW", "Leukemia", "Lymphography", "M-of-n",
    "Parkinsons", "Sonar", "SpectEW", "TicTacToe", "Vote", "WDBC",
    "WaveformEW", "Zoo",
]

                                         
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
