"""Feature-selection stability index (Diem_yeu_RG-SCSO.md §2.2) — RG-SCSO's
central contribution is parsimony at preserved accuracy, but the paper never
quantifies whether the SELECTED SUBSET ITSELF is consistent across independent
runs, only its size and downstream accuracy. A method that selects a
different, equally-small, equally-accurate subset on every run is a much
weaker practical claim than one that converges on largely the same features.

METHOD: Nogueira, Sechidis & Brown (2018), "On the Stability of Feature
Selection Algorithms," JMLR 18(174):1-54 — the standard generalization of the
Kuncheva (2007) consistency index to VARIABLE subset size (RG-SCSO's subset
size is not fixed across runs, so the classical equal-size Kuncheva formula
does not directly apply; Nogueira's Phi is the correctly-cited modern
replacement, reducing to Kuncheva's index in the equal-size case).

Given a binary indicator matrix Z in {0,1}^(M x d) (M runs, d features),
p_j = mean_i Z[i,j] (selection frequency of feature j), k_bar = mean subset
size:

    Phi = 1 - [ (1/d) * sum_j (M/(M-1)) * p_j*(1-p_j) ]
              / [ (k_bar/d) * (1 - k_bar/d) ]

Phi in [-1, 1] (typically); 1 = identical subset every run, 0 = no more
consistent than selecting k_bar features uniformly at random each run.

SCOPE: same 5-dataset representative subset used throughout this paper's
robustness/ablation pilots (Zoo, Sonar, WDBC, ColonCancer, Leukemia), 30
independent runs per algorithm (matching the main study's run count), for
RG-SCSO (the deployed configuration), SCSO (no relevance signal, in-family
reference), and AOA (cross-family reference) — masks are saved per run
(compact selected-index string, not a full bit matrix, since Leukemia has
3571 features) so Phi can be computed after the fact.

Output: experiments/results_stability/stability_masks.csv (raw masks)
        experiments/results_stability/stability_index_results.csv (Phi per
        algorithm x dataset)
Run:    .venv/bin/python -m src.feature_selection.run_stability_index [--smoke]
        [--datasets ...] [--runs N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, NUM_INDEPENDENT_RUNS, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso_fs_baselines import BinarySCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_stability")
MASKS_CSV = os.path.join(OUTPUT_DIR, "stability_masks.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "stability_index_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
ALGOS = ["RG-SCSO", "SCSO", "AOA"]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(MASKS_CSV):
        return set()
    df = pd.read_csv(MASKS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    algo, ds, run_id = task["algorithm"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    if algo == "RG-SCSO":
        result = RGSCSO(
            obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
            pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
            X=X, y=y, eval_mask=eval_mask,
        ).optimize()
        mask = result["best_mask"]
    elif algo == "SCSO":
        result = BinarySCSO(obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE,
                             MAX_ITERATION, seed, eval_mask=eval_mask,
                             transfer_kind="s", use_obl=False).optimize()
        mask = result["best_mask"]
    else:  # AOA — mealpy baseline, thresholded like elsewhere in this project
        from src.feature_selection.transfer_function import binarize_threshold
        result = run_mealpy_baseline(algo, obj_func, dim=dim, lb=SEARCH_LB,
                                      ub=SEARCH_UB, pop_size=POPULATION_SIZE,
                                      max_iter=MAX_ITERATION, seed=seed)
        mask = binarize_threshold(result["best_solution"])

    indices = ";".join(str(i) for i in np.flatnonzero(mask))
    return {
        "algorithm": algo, "dataset": ds, "run_id": run_id,
        "n_total_features": dim, "n_selected_features": int(mask.sum()),
        "selected_indices": indices,
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(MASKS_CSV)
    df.to_csv(MASKS_CSV, mode="a", header=header, index=False)


def nogueira_phi(masks: list[set[int]], d: int) -> float:
    """Nogueira et al. (2018) stability measure Phi for M runs' selected-index
    sets (variable size), d = total feature count. See module docstring."""
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


def _compute_stability(masks_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, ds), g in masks_df.groupby(["algorithm", "dataset"]):
        d = int(g["n_total_features"].iloc[0])
        sets = [
            set(int(i) for i in s.split(";")) if isinstance(s, str) and s else set()
            for s in g["selected_indices"]
        ]
        phi = nogueira_phi(sets, d)
        rows.append({
            "algorithm": algo, "dataset": ds, "n_runs": len(sets),
            "n_total_features": d,
            "mean_n_selected": float(np.mean([len(s) for s in sets])),
            "nogueira_phi": phi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, RG-SCSO — kiểm tra wiring + Phi trên self-pair giả.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, algos, n_runs, workers = ["Zoo"], ["RG-SCSO"], 2, 1
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
    print(f"Stability: {len(algos)} algo x {len(datasets)} dataset x {n_runs} "
          f"run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

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
        for fut in tqdm(as_completed(futures), total=len(futures), desc="stability"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    masks_df = pd.read_csv(MASKS_CSV)
    stability_df = _compute_stability(masks_df)
    stability_df.to_csv(RESULTS_CSV, index=False)
    print(stability_df.to_string(index=False))
    print(f"Xong. Masks tại {MASKS_CSV}, Phi tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
