"""Nested cross-validation pilot (Q1 review Priority 6) — chuẩn mạnh hơn cho
một wrapper metaheuristic so với hold-out 80/20 đơn lẻ hiện tại: outer CV ->
inner FS optimization (search + fitness) -> outer test, lặp lại trên MỌI outer
fold thay vì một lần chia duy nhất, để loại trừ khả năng kết quả held-out phụ
thuộc vào MAY RỦI của đúng 1 lần chia 80/20.

THIẾT KẾ (pilot, KHÔNG thay thế held-out 80/20 chính — bổ sung xác nhận):
    outer : StratifiedKFold k=5 (mỗi fold lượt lần lượt làm test, 4 fold còn
            lại làm train — KHÔNG chỉ 1 lần 80/20).
    inner : y hệt bảng chính — fitness = KNN 5-fold CV TRÊN CHÍNH outer-train
            (RG-SCSO: prior MI cũng CHỈ tính trên outer-train, không leak).
    scope : 3 dataset đại diện (Zoo thấp chiều, WDBC trung bình, ColonCancer
            gene-expression cao chiều) x RG-SCSO/SCSO/AOA x 5 run — quy mô
            PILOT có chủ đích (mỗi run đã tốn 5x compute so với held-out 1-split
            vì phải chạy search TRÊN MỌI outer fold); mở rộng ra 18 dataset x
            30 run là bước tiếp theo nếu pilot cho tín hiệu nhất quán với
            held-out 80/20 đã có.

Output: experiments/results_fs_nested_cv/fs_nested_cv_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_nested_cv [--smoke]
        [--datasets ...] [--runs N] [--outer-k K]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    KNN_NEIGHBORS,
    MAX_ITERATION,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function
from src.feature_selection.transfer_function import binarize_threshold

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_nested_cv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_nested_cv_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
DEFAULT_DATASETS = ["Zoo", "WDBC", "ColonCancer"]
DEFAULT_OUTER_K = 5
DEFAULT_RUNS = 5
ALGOS = ["RG-SCSO", "SCSO", "AOA"]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _outer_test_acc(mask, x_train, y_train, x_test, y_test) -> float:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return 0.0
    pipe = Pipeline([("scaler", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))])
    pipe.fit(x_train[:, selected], y_train)
    return float(pipe.score(x_test[:, selected], y_test))


def _existing_keys() -> set[tuple[str, str, int, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"], df["outer_fold"]))


def _run_single(task: dict) -> dict:
    algo, ds, run_id, outer_k = task["algorithm"], task["dataset"], task["run_id"], task["outer_k"]
    X, y = _load(ds)
    seed = RANDOM_SEED_BASE + run_id
    skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    fold_accs = []
    fold_nfeats = []
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        x_train, x_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]
        dim = x_train.shape[1]
        obj_func = make_fitness_function(x_train, y_train, seed=seed)

        if algo == "RG-SCSO":
            def eval_mask(mask: np.ndarray) -> float:
                return evaluate_binary_mask(mask, x_train, y_train, seed=seed)["fitness"]
            result = RGSCSO(
                obj_func=obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
                pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
                X=x_train, y=y_train, eval_mask=eval_mask,
            ).optimize()
            mask = result["best_mask"]
        elif algo == "SCSO":
            result = SCSO(obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE,
                           MAX_ITERATION, seed).optimize()
            mask = binarize_threshold(result["best_solution"])
        else:
            result = run_mealpy_baseline(algo, obj_func, dim=dim, lb=SEARCH_LB,
                                          ub=SEARCH_UB, pop_size=POPULATION_SIZE,
                                          max_iter=MAX_ITERATION, seed=seed)
            mask = binarize_threshold(result["best_solution"])

        acc = _outer_test_acc(mask, x_train, y_train, x_test, y_test)
        fold_accs.append(acc)
        fold_nfeats.append(int(mask.sum()))

    return {
        "algorithm": algo, "dataset": ds, "run_id": run_id, "outer_fold": outer_k,
        "mean_nested_cv_accuracy": float(np.mean(fold_accs)),
        "std_nested_cv_accuracy": float(np.std(fold_accs)),
        "mean_n_selected_features": float(np.mean(fold_nfeats)),
        "n_total_features": X.shape[1],
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, RG-SCSO, outer-k=3 — kiểm tra wiring.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--outer-k", type=int, default=DEFAULT_OUTER_K)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, algos, n_runs, outer_k, workers = ["Zoo"], ["RG-SCSO"], 1, 3, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        algos = ALGOS
        n_runs, outer_k, workers = args.runs, args.outer_k, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": a, "dataset": ds, "run_id": r, "outer_k": outer_k}
        for a in algos for ds in datasets for r in range(n_runs)
        if (a, ds, r, outer_k) not in done
    ]
    total = len(algos) * len(datasets) * n_runs
    print(f"Nested CV: {len(algos)} algo x {len(datasets)} dataset x {n_runs} run "
          f"x {outer_k}-fold outer = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="nested-cv"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
