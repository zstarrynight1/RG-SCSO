
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function
from src.feature_selection.transfer_function import binarize_threshold

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_convergence")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "convergence_curves.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "WDBC", "ColonCancer"]
DEFAULT_RUNS = 5
ALGOS = ["RG-SCSO", "SCSO", "AOA"]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> list[dict]:
    algo, ds, run_id = task["algorithm"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    if algo == "RG-SCSO":
        def eval_mask(mask: np.ndarray) -> float:
            return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]
        result = RGSCSO(
            obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
            pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
            X=X, y=y, eval_mask=eval_mask,
        ).optimize()
    elif algo == "SCSO":
        result = SCSO(obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE,
                       MAX_ITERATION, seed).optimize()
    else:                         
        result = run_mealpy_baseline(algo, obj_func, dim=dim, lb=SEARCH_LB,
                                      ub=SEARCH_UB, pop_size=POPULATION_SIZE,
                                      max_iter=MAX_ITERATION, seed=seed)

    curve = result["convergence_curve"]


    nfe_curve = result.get("nfe_curve")
    if nfe_curve is None:
        nfe_curve = [POPULATION_SIZE * (it + 1) for it in range(len(curve))]
    return [
        {"algorithm": algo, "dataset": ds, "run_id": run_id, "iteration": it,
         "nfe": int(nfe), "fitness": float(f)}
        for it, (f, nfe) in enumerate(zip(curve, nfe_curve))
    ]


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, RG-SCSO — kiểm tra wiring, không phải kết quả.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, algos, n_runs, workers = ["Zoo"], ["RG-SCSO"], 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        algos = ALGOS
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": a, "dataset": ds, "run_id": r}
        for a in algos for ds in datasets for r in range(n_runs)
        if (a, ds, r) not in done
    ]
    total = len(algos) * len(datasets) * n_runs
    print(f"Convergence curves: {len(algos)} algo x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = []
        for t in tasks:
            rows.extend(_run_single(t))
        df = pd.DataFrame(rows)
        print(f"curve length: {len(df)}, final fitness: {df['fitness'].iloc[-1]:.4f}")
        print("SMOKE OK — wiring chạy, convergence_curve hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="convergence"):
            try:
                _append_rows(fut.result())
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
