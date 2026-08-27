"""Threshold sensitivity study (RG-SCSO_MASTER_FINAL_COMPLETE.md, 5 most-
important-items list, item 1) — RG-SCSO's RMS binarization uses tau=0.5 as
the decision threshold separating "preferred" from "disfavored" bits
(b*_j = 1 if rho_j > tau else 0). The paper already discloses tau=0.5 is a
convenience, not a theoretically grounded neutral point, but never actually
SWEEPS it. This harness runs RG-SCSO at tau in {0.4, 0.5, 0.6} and reports
accuracy, feature count, and feature-selection stability (Nogueira Phi, same
method as run_stability_index.py) at each value, so the disclosed caveat is
backed by real data rather than left as an unquantified hedge.

tau=0.5 is EXACTLY the existing main-study RG-SCSO configuration (verified:
binarize_relevance(..., threshold=0.5) reduces algebraically to the original
un-parameterized formula, confirmed via a direct before/after equality check
before this harness was written) -- no existing result is invalidated by this
addition, tau is a strictly additive new capability.

SCOPE: same 5-dataset representative subset used throughout this paper's
pilots (Zoo, Sonar, WDBC, ColonCancer, Leukemia), 30 independent runs per
(tau, dataset) cell, matching the main protocol (pop=30, iter=500,
seed=BASE+run_id).

Output: experiments/results_threshold/threshold_masks.csv (raw masks, for
        Phi computation)
        experiments/results_threshold/threshold_sensitivity_results.csv
        (accuracy, n_selected_features, Phi per tau x dataset)
Run:    .venv/bin/python -m src.feature_selection.run_threshold_sensitivity
        [--smoke] [--datasets ...] [--runs N] [--taus 0.4,0.5,0.6]
"""

from __future__ import annotations

import os

# Phải set TRƯỚC khi import numpy — tránh deadlock ProcessPoolExecutor +
# threaded BLAS đã gặp trước đó trong phiên này (macOS fork()-sau-khi-có-
# thread). Áp dụng chủ động ngay từ đầu thay vì chờ phát hiện lại bug.
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
OUTPUT_DIR = os.path.join("experiments", "results_threshold")
MASKS_CSV = os.path.join(OUTPUT_DIR, "threshold_masks.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "threshold_sensitivity_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
DEFAULT_TAUS = [0.4, 0.5, 0.6]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[float, str, int]]:
    if not os.path.exists(MASKS_CSV):
        return set()
    df = pd.read_csv(MASKS_CSV)
    return set(zip(df["tau"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    tau, ds, run_id = task["tau"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    result = RGSCSO(
        obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
        pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
        X=X, y=y, eval_mask=eval_mask, tau=tau,
    ).optimize()
    mask = result["best_mask"]
    final = evaluate_binary_mask(mask, X, y, seed=seed)
    indices = ";".join(str(i) for i in np.flatnonzero(mask))
    return {
        "tau": tau, "dataset": ds, "run_id": run_id,
        "fitness": final["fitness"], "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "selected_indices": indices,
        "runtime_seconds": result["runtime"],
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(MASKS_CSV)
    df.to_csv(MASKS_CSV, mode="a", header=header, index=False)


def nogueira_phi(masks: list[set[int]], d: int) -> float:
    """Cùng công thức Nogueira et al. (2018) đã dùng ở run_stability_index.py
    — xem file đó để biết chi tiết diễn giải/nguồn gốc."""
    m = len(masks)
    if m < 2:
        return float("nan")
    counts = np.zeros(d, dtype=float)
    for s in masks:
        for j in s:
            counts[j] += 1.0
    p = counts / m
    k_bar = float(np.mean([len(s) for s in masks]))
    numerator = (1.0 / d) * np.sum((m / (m - 1)) * p * (1.0 - p))
    denom = (k_bar / d) * (1.0 - k_bar / d)
    if denom <= 0:
        return float("nan")
    return 1.0 - numerator / denom


def _summarize(masks_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tau, ds), g in masks_df.groupby(["tau", "dataset"]):
        d = int(g["n_total_features"].iloc[0])
        sets = [
            set(int(i) for i in s.split(";")) if isinstance(s, str) and s else set()
            for s in g["selected_indices"]
        ]
        phi = nogueira_phi(sets, d)
        rows.append({
            "tau": tau, "dataset": ds, "n_runs": len(sets),
            "n_total_features": d,
            "mean_accuracy": float(g["accuracy"].mean()),
            "mean_n_selected": float(np.mean([len(s) for s in sets])),
            "nogueira_phi": phi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, tau=0.5 — kiểm tra wiring + Phi trên self-pair giả.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    ap.add_argument("--taus", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, taus, n_runs, workers = ["Zoo"], [0.5], 2, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        taus = [float(t) for t in args.taus.split(",")] if args.taus else DEFAULT_TAUS
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"tau": t, "dataset": ds, "run_id": r}
        for t in taus for ds in datasets for r in range(n_runs)
        if (t, ds, r) not in done
    ]
    total = len(taus) * len(datasets) * n_runs
    print(f"Threshold sensitivity: {len(taus)} tau x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        df = pd.DataFrame(rows)
        print(df.to_string())
        phi_test = nogueira_phi(
            [set(int(i) for i in r.split(";")) if r else set()
             for r in df["selected_indices"]],
            int(df["n_total_features"].iloc[0]),
        )
        print(f"\nSanity Phi (2 runs, same dataset) = {phi_test:.4f} (finite, no crash)")
        print("SMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="threshold"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] tau={t['tau']} x {t['dataset']} x run{t['run_id']}: {exc}")

    masks_df = pd.read_csv(MASKS_CSV)
    summary_df = _summarize(masks_df)
    summary_df.to_csv(RESULTS_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"Xong. Masks tại {MASKS_CSV}, summary tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
