
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

from config import MAX_ITERATION, NUM_INDEPENDENT_RUNS, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.rg_scso import RGSCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_lambda")
RAW_CSV = os.path.join(OUTPUT_DIR, "lambda_raw.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "lambda_sensitivity_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
DEFAULT_BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[float, str, int]]:
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    return set(zip(df["beta"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    beta, ds, run_id = task["beta"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed, beta=beta)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed, beta=beta)["fitness"]

    result = RGSCSO(
        obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
        pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
        X=X, y=y, eval_mask=eval_mask,
    ).optimize()
    mask = result["best_mask"]
    final = evaluate_binary_mask(mask, X, y, seed=seed, beta=beta)
    return {
        "beta": beta, "dataset": ds, "run_id": run_id,
        "fitness": final["fitness"], "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "runtime_seconds": result["runtime"],
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(RAW_CSV)
    df.to_csv(RAW_CSV, mode="a", header=header, index=False)


def _summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (beta, ds), g in raw_df.groupby(["beta", "dataset"]):
        rows.append({
            "beta": beta, "dataset": ds, "n_runs": len(g),
            "n_total_features": int(g["n_total_features"].iloc[0]),
            "mean_accuracy": float(g["accuracy"].mean()),
            "mean_n_selected": float(g["n_selected_features"].mean()),
            "std_accuracy": float(g["accuracy"].std()),
            "std_n_selected": float(g["n_selected_features"].std()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, beta=0.01 -- kiểm tra wiring, không phải kết quả.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    ap.add_argument("--betas", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, betas, n_runs, workers = ["Zoo"], [0.01], 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        betas = [float(b) for b in args.betas.split(",")] if args.betas else DEFAULT_BETAS
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"beta": b, "dataset": ds, "run_id": r}
        for b in betas for ds in datasets for r in range(n_runs)
        if (b, ds, r) not in done
    ]
    total = len(betas) * len(datasets) * n_runs
    print(f"Lambda sensitivity: {len(betas)} beta x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        df = pd.DataFrame(rows)
        print(df.to_string())
        print("SMOKE OK -- wiring chạy, kết quả hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="lambda"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] beta={t['beta']} x {t['dataset']} x run{t['run_id']}: {exc}")

    raw_df = pd.read_csv(RAW_CSV)
    summary_df = _summarize(raw_df)
    summary_df.to_csv(RESULTS_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"Xong. Raw tại {RAW_CSV}, summary tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
