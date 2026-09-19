
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from config import (
    KNN_NEIGHBORS,
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso_fs_baselines import BinarySCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_robustness")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_robustness_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0


DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
DEFAULT_WRAPPERS = ["KNN", "SVM", "RF"]

                                                                                       
ALGOS = {
    "RG-SCSO-MI": {"kind": "rgscso", "prior": "mi"},
    "RG-SCSO-ReliefF": {"kind": "rgscso", "prior": "relieff"},
    "bSCSO": {"kind": "bscso", "prior": None},
}


def _make_clf(wrapper: str, seed: int):
    if wrapper == "KNN":
        return KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    if wrapper == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    if wrapper == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    raise ValueError(f"wrapper không hỗ trợ: {wrapper}")


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _existing_keys() -> set[tuple[str, str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["wrapper"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    algo, wrapper, ds, run_id = (
        task["algorithm"], task["wrapper"], task["dataset"], task["run_id"]
    )
    spec = ALGOS[algo]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id

    def clf_factory():
        return _make_clf(wrapper, seed)

    obj_func = make_fitness_function(X, y, seed=seed, clf_factory=clf_factory)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed, clf_factory=clf_factory)["fitness"]

    if spec["kind"] == "rgscso":
        result = RGSCSO(
            obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
            pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
            X=X, y=y, eval_mask=eval_mask, prior_method=spec["prior"],
        ).optimize()
    else:                                                               
        result = BinarySCSO(
            obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
            eval_mask=eval_mask, transfer_kind="s", use_obl=False,
        ).optimize()

    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed, clf_factory=clf_factory)
    return {
        "algorithm": algo, "wrapper": wrapper, "prior": spec["prior"] or "none",
        "dataset": ds, "run_id": run_id,
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
                    help="1 run, Zoo, KNN, 1 algo — kiểm tra wiring (KHÔNG phải kết quả).")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--wrappers", type=str, default=None)
    ap.add_argument("--algos", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, wrappers, algos, n_runs, workers = (
            ["Zoo"], ["SVM"], ["RG-SCSO-ReliefF"], 1, 1
        )
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        wrappers = args.wrappers.split(",") if args.wrappers else DEFAULT_WRAPPERS
        algos = args.algos.split(",") if args.algos else list(ALGOS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": a, "wrapper": w, "dataset": ds, "run_id": r}
        for a in algos for w in wrappers for ds in datasets for r in range(n_runs)
        if (a, w, ds, r) not in done
    ]
    total = len(algos) * len(wrappers) * len(datasets) * n_runs
    print(f"Robustness: {len(algos)} algo x {len(wrappers)} wrapper x "
          f"{len(datasets)} dataset x {n_runs} run = {total} task "
          f"({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="robustness"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['wrapper']} x {t['dataset']} "
                      f"x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
