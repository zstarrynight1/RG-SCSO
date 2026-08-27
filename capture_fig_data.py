"""Tái tạo convergence-curve + selected-mask cho hình 3-4 (KHÔNG lưu trong
fs_results.csv). Chạy lại đúng seed 42+run_id (deterministic, spot-check R4 đã
chứng minh bit-for-bit) — CHỈ để VẼ, không sinh số báo cáo mới.

Sinh figures/fig_capture.npz:
    curves[dataset][algo]  : ndarray (n_runs, T)  mean-best-fitness theo iter.
    overlap[dataset][algo] : ndarray (n_runs,)    precision@|S| so với top-MI.
    conv_datasets, mech_datasets, n_runs (metadata cho make_figures.py).

Chạy: .venv/bin/python capture_fig_data.py   (nền, ~50 phút, 8 worker)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function
from src.feature_selection.relevance import relevance_prior
from src.feature_selection.transfer_function import binarize_threshold

PROCESSED_DIR = os.path.join("data", "processed")
OUT = os.path.join("figures", "fig_capture.npz")
SEARCH_LB, SEARCH_UB = -1.0, 1.0

# Low-dim + high-dim cho convergence; 2 gene-set cho mechanism (relevance-guided
# vs agnostic). Chọn dataset ÍT MẪU để CV rẻ (Zoo/Colon/Leukemia đều <110 mẫu).
CONV_DATASETS = ["Zoo", "ColonCancer"]
MECH_DATASETS = ["ColonCancer", "Leukemia"]
CONV_ALGOS = ["RG-SCSO", "SCSO", "AOA", "RIME"]
MECH_ALGOS = ["RG-SCSO", "SCSO"]
N_RUNS = 30


def _load(name: str):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(float), df["label"].to_numpy()


def _capture_one(algo: str, dataset: str, run_id: int) -> dict:
    """1 run: trả convergence_curve + chỉ số feature được chọn (selected_idx)."""
    X, y = _load(dataset)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj = make_fitness_function(X, y, seed=seed)

    if algo == "RG-SCSO":
        def em(mask):
            return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]
        res = RGSCSO(obj, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION,
                     seed, X=X, y=y, eval_mask=em).optimize()
        mask = np.asarray(res["best_mask"], dtype=int)
    elif algo == "SCSO":
        res = SCSO(obj, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE,
                   MAX_ITERATION, seed).optimize()
        mask = binarize_threshold(res["best_solution"])
    else:
        res = run_mealpy_baseline(algo, obj, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
                                  pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION,
                                  seed=seed)
        mask = binarize_threshold(res["best_solution"])

    return {
        "algo": algo, "dataset": dataset, "run_id": run_id,
        "curve": np.asarray(res["convergence_curve"], dtype=float),
        "sel": np.flatnonzero(mask).astype(int),
    }


def _tasks():
    need = {ds: set(CONV_ALGOS) for ds in CONV_DATASETS}
    for ds in MECH_DATASETS:
        need.setdefault(ds, set()).update(MECH_ALGOS)
    return [(a, ds, r) for ds, algos in need.items()
            for a in sorted(algos) for r in range(N_RUNS)]


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    tasks = _tasks()
    print(f"Tái tạo {len(tasks)} run ({len(set(t[1] for t in tasks))} dataset) "
          f"cho convergence + mechanism...")

    # gom kết quả: raw[(algo,ds)] = {run_id: {curve, sel}}
    raw: dict = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = {ex.submit(_capture_one, a, ds, r): (a, ds, r) for a, ds, r in tasks}
        for f in tqdm(as_completed(futs), total=len(futs), desc="capture"):
            a, ds, r = futs[f]
            try:
                out = f.result()
                raw.setdefault((a, ds), {})[r] = out
            except Exception as exc:  # noqa: BLE001
                print(f"[LỖI] {a} x {ds} x run{r}: {exc}")

    # --- convergence: pad curve về đúng MAX_ITERATION, stack theo run ---
    curves: dict = {}
    for ds in CONV_DATASETS:
        curves[ds] = {}
        for a in CONV_ALGOS:
            runs = raw.get((a, ds), {})
            if not runs:
                continue
            arr = np.zeros((len(runs), MAX_ITERATION))
            for i, r in enumerate(sorted(runs)):
                c = runs[r]["curve"]
                if len(c) < MAX_ITERATION:
                    c = np.concatenate([c, np.full(MAX_ITERATION - len(c), c[-1])])
                arr[i] = c[:MAX_ITERATION]
            curves[ds][a] = arr

    # --- mechanism: precision@|S| = |selected ∩ top-|S| MI| / |S| ---
    # top-MI set = ranking relevance_prior(seed=42) — chuẩn "độ liên quan gốc".
    overlap: dict = {}
    for ds in MECH_DATASETS:
        X, y = _load(ds)
        mi = relevance_prior(X, y, RANDOM_SEED_BASE)   # seed cố định = tham chiếu
        order = np.argsort(-mi)                         # feature MI cao → thấp
        overlap[ds] = {}
        for a in MECH_ALGOS:
            runs = raw.get((a, ds), {})
            if not runs:
                continue
            vals = []
            for r in sorted(runs):
                sel = runs[r]["sel"]
                if len(sel) == 0:
                    vals.append(0.0)
                    continue
                topk = set(order[: len(sel)].tolist())
                vals.append(len(set(sel.tolist()) & topk) / len(sel))
            overlap[ds][a] = np.asarray(vals, dtype=float)

    np.savez(OUT, curves=np.array(curves, dtype=object),
             overlap=np.array(overlap, dtype=object),
             conv_datasets=np.array(CONV_DATASETS),
             mech_datasets=np.array(MECH_DATASETS),
             n_runs=N_RUNS)
    print(f"Đã ghi {OUT}")
    # tóm tắt nhanh mechanism để kiểm tra ngay
    for ds in MECH_DATASETS:
        line = ", ".join(f"{a} {overlap[ds][a].mean()*100:.1f}%"
                         for a in MECH_ALGOS if a in overlap[ds])
        print(f"  overlap top-MI [{ds}]: {line}")


if __name__ == "__main__":
    main()
