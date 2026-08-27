"""NFE-matched random-probe control (RG-SCSO_MASTER_FINAL_COMPLETE.md, "NFE
control" / 5 most-important-items #2) — isolates whether UMR's (C3) benefit
comes from TARGETING the K features nearest the RMS decision boundary
(uncertain -> most likely to flip a wrong bit), or simply from spending K
extra evaluations of local search ANYWHERE, at the same NFE cost.

The existing ablation ("NoUMR", already in experiments/results_fs/
fs_ablation_results.csv) removes C3 entirely -- it confounds "loses targeting"
with "loses the extra evaluations altogether". RGSCSORandomProbe (see
src/algorithms/rg_scso_random_probe.py) keeps the exact same NFE budget and
greedy-accept-if-better logic, but picks its K probe features uniformly at
random instead of by relevance-uncertainty -- the genuine NFE-matched control
this comparison needs.

Three-way comparison per dataset (all under the SAME protocol as the main
ablation): RG-SCSO (targeted UMR, final deployed config) vs RG-SCSO-RandomProbe
(this new control) vs -UMR (existing ablation CSV, reused rather than rerun --
same seeds, same protocol, already verified data). If targeted beats
RandomProbe by a margin similar to how it beats -UMR entirely, targeting is
what matters. If RandomProbe performs comparably to targeted UMR, that is an
honest, disclosable null result on the value of TARGETING specifically (not on
UMR/extra-search-effort as a whole) -- report whichever the data shows.

SCOPE: same 5-dataset representative set as every other pilot this session
(Zoo, Sonar, WDBC, ColonCancer, Leukemia), 30 runs, matching main protocol.

Output: experiments/results_nfe_control/nfe_control_results.csv
Run:    .venv/bin/python -m src.feature_selection.run_nfe_matched_control
        [--smoke] [--datasets ...] [--runs N]
"""

from __future__ import annotations

import os

# Phải set TRƯỚC khi import numpy — tránh deadlock ProcessPoolExecutor +
# threaded BLAS đã gặp trước đó trong phiên này.
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
from src.algorithms.rg_scso_random_probe import RGSCSORandomProbe
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_nfe_control")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "nfe_control_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
# "RG-SCSO" ở đây trùng đúng cấu hình Full-2-component đã có (RMS+UMR targeted)
# -- chạy lại độc lập (không tái dùng CSV cũ) để cùng seed/protocol/timestamp
# với RandomProbe, tránh mọi khác biệt môi trường lặt vặt làm nhiễu so sánh.
CONFIGS = ["RG-SCSO", "RG-SCSO-RandomProbe"]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["config_name"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    config, ds, run_id = task["config"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    cls = RGSCSO if config == "RG-SCSO" else RGSCSORandomProbe
    result = cls(
        obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
        pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
        X=X, y=y, eval_mask=eval_mask,
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    return {
        "config_name": config, "dataset": ds, "run_id": run_id,
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
                     help="1 run, Zoo, cả 2 config — kiểm tra wiring.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, configs, n_runs, workers = ["Zoo"], CONFIGS, 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        configs = CONFIGS
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"config": c, "dataset": ds, "run_id": r}
        for c in configs for ds in datasets for r in range(n_runs)
        if (c, ds, r) not in done
    ]
    total = len(configs) * len(datasets) * n_runs
    print(f"NFE-matched control: {len(configs)} config x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="nfe-control"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['config']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
