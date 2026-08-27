"""Fold-honest held-out generalization run cho RG-SCSO (R3b).

ĐỘNG CƠ (2026-07-08): confound audit phát hiện `relevance_prior` tính MI trên
TOÀN BỘ (X, y) — gồm nhãn test fold — nên chỉ RG-SCSO hưởng một leak transductive
(mean Spearman ρ_full vs ρ_train = 0.75, nặng trên gene-set). Pilot fold-honest
(80/20 held-out, budget-matched) cho RG-SCSO 7 WIN / 0 tie / 1 LOSS vs SCSO ⇒
win THẬT, không phải artifact. Run này sản xuất BẢNG GENERALIZATION chính thức để:
    (a) chứng minh không leak (MI prior chỉ tính trên train-80),
    (b) trả lời reviewer "no generalization study",
    (c) cho effect size THỰC TẾ (held-out) bên cạnh in-sample CV.

GIAO THỨC (khóa trước khi chạy — pre-registration, spec 8.1/4.2):
    Với mỗi (algorithm, dataset, run_id):
      seed = RANDOM_SEED_BASE + run_id
      1. Outer split 80/20 stratified (random_state=seed) -> (Xtr, ytr), (Xte, yte).
      2. TÌM KIẾM + FITNESS chỉ trên (Xtr, ytr):
           - fitness = KNN 5-fold CV TRONG train-80 (StandardScaler fit per-fold).
           - RG-SCSO: prior relevance ρ_static = MI(Xtr, ytr) — KHÔNG nhìn held-out.
      3. Chốt subset (RG-SCSO: best_mask; còn lại: binarize_threshold(best_solution)).
      4. BÁO CÁO: accuracy trên HELD-OUT-20 (fit KNN+scaler trên train-80[selected],
         score trên held-out-20[selected]). Metric of record = heldout_accuracy.
    Budget khớp baseline (pop_size × max_iter đúng config.py). Không đổi thuật toán,
    dataset, seed, kiểm định sau khi thấy số. Giữ song song bảng in-sample R3 gốc
    làm main results; bảng này là generalization validation.

Output (append từng dòng, resumable như runner gốc):
    experiments/results_fs_heldout/fs_heldout_results.csv
    experiments/results_fs_heldout/fs_heldout_summary.csv

Chạy:  caffeinate -i python -m src.feature_selection.run_fs_heldout
       python -m src.feature_selection.run_fs_heldout --summary-only
       python -m src.feature_selection.run_fs_heldout --smoke   # smoke test nhanh
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    KNN_NEIGHBORS,
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function
from src.feature_selection.transfer_function import binarize_threshold

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_heldout")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_heldout_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "fs_heldout_summary.csv")

SEARCH_LB = -1.0
SEARCH_UB = 1.0
TEST_SIZE = 0.2

# Bộ thuật toán KHỚP main results R3 (bulletproof so sánh cùng đối thủ).
MEALPY_BASELINES = ["PSO", "GWO", "AOA", "COA", "RIME"]
ALL_ALGORITHMS = ["RG-SCSO", "SCSO"] + MEALPY_BASELINES

FIELDNAMES = [
    "algorithm",
    "dataset",
    "run_id",
    "heldout_accuracy",
    "train_cv_fitness",
    "n_selected_features",
    "n_total_features",
    "runtime_seconds",
]


def get_dataset_names() -> list[str]:
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    X = df.drop(columns=["label"]).to_numpy(dtype=float)
    y = df["label"].to_numpy()
    return X, y


def _heldout_accuracy(
    mask: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_neighbors: int = KNN_NEIGHBORS,
) -> float:
    """Accuracy trên held-out: fit StandardScaler+KNN trên train-80[selected],
    score trên held-out-20[selected]. Không rò rỉ — scaler/KNN chỉ thấy train."""
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return 0.0
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k_neighbors))]
    )
    pipe.fit(x_train[:, selected], y_train)
    return float(pipe.score(x_test[:, selected], y_test))


def _run_single_task(algorithm: str, dataset: str, run_id: int) -> dict:
    X, y = _load_dataset(dataset)
    seed = RANDOM_SEED_BASE + run_id
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    dim = x_train.shape[1]

    # Fitness + search CHỈ trên train-80 (fold-honest).
    obj_func = make_fitness_function(x_train, y_train, seed=seed)

    if algorithm == "RG-SCSO":
        def eval_mask(mask: np.ndarray) -> float:
            return evaluate_binary_mask(mask, x_train, y_train, seed=seed)["fitness"]

        result = RGSCSO(
            obj_func=obj_func,
            dim=dim,
            lb=SEARCH_LB,
            ub=SEARCH_UB,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATION,
            seed=seed,
            X=x_train,  # prior MI chỉ nhìn train-80
            y=y_train,
            eval_mask=eval_mask,
        ).optimize()
        mask = result["best_mask"]
    elif algorithm == "SCSO":
        result = SCSO(
            obj_func=obj_func,
            dim=dim,
            lb=SEARCH_LB,
            ub=SEARCH_UB,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATION,
            seed=seed,
        ).optimize()
        mask = binarize_threshold(result["best_solution"])
    else:
        result = run_mealpy_baseline(
            algorithm,
            obj_func,
            dim=dim,
            lb=SEARCH_LB,
            ub=SEARCH_UB,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATION,
            seed=seed,
        )
        mask = binarize_threshold(result["best_solution"])

    train_cv = evaluate_binary_mask(mask, x_train, y_train, seed=seed)
    heldout_acc = _heldout_accuracy(mask, x_train, y_train, x_test, y_test)
    return {
        "algorithm": algorithm,
        "dataset": dataset,
        "run_id": run_id,
        "heldout_accuracy": heldout_acc,
        "train_cv_fitness": train_cv["fitness"],
        "n_selected_features": train_cv["n_selected_features"],
        "n_total_features": train_cv["n_total_features"],
        "runtime_seconds": result["runtime"],
    }


def _existing_completed_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return {(row.algorithm, row.dataset, int(row.run_id)) for row in df.itertuples()}


def write_summary() -> None:
    df = pd.read_csv(RESULTS_CSV)
    rows = []
    for (algorithm, dataset), group in df.groupby(["algorithm", "dataset"]):
        for metric in ["heldout_accuracy", "train_cv_fitness", "n_selected_features"]:
            values = group[metric].tolist()
            rows.append(
                {
                    "algorithm": algorithm,
                    "dataset": dataset,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "best": max(values) if metric != "train_cv_fitness" else min(values),
                    "worst": min(values) if metric != "train_cv_fitness" else max(values),
                }
            )
    pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric"]).to_csv(
        SUMMARY_CSV, index=False
    )
    print(f"Đã ghi {SUMMARY_CSV} ({len(rows)} dòng).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Smoke test nhanh (1 ds, 1 run).")
    parser.add_argument(
        "--algos",
        default=None,
        help="Danh sách thuật toán chạy (phẩy ngăn cách) để chạy phân tầng, "
        "ví dụ 'RG-SCSO,SCSO,AOA'. Bỏ trống = cả 7. Runner resumable nên chạy "
        "tầng sau bổ sung vào cùng CSV, tự bỏ task đã xong.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.summary_only:
        write_summary()
        return

    dataset_names = get_dataset_names()
    num_runs = NUM_INDEPENDENT_RUNS

    algos = ALL_ALGORITHMS
    if args.algos:
        requested = [a.strip() for a in args.algos.split(",") if a.strip()]
        unknown = [a for a in requested if a not in ALL_ALGORITHMS]
        if unknown:
            raise SystemExit(f"Thuật toán không hợp lệ: {unknown}. Hợp lệ: {ALL_ALGORITHMS}")
        algos = requested

    if args.smoke:
        dataset_names = dataset_names[:1]
        num_runs = 1
        print(f"[SMOKE] {dataset_names} x {len(algos)} algo x {num_runs} run")

    all_tasks = [
        (algo, ds, run_id)
        for algo in algos
        for ds in dataset_names
        for run_id in range(num_runs)
    ]
    done = _existing_completed_keys()
    tasks = [t for t in all_tasks if t not in done]

    print(
        f"Tổng {len(all_tasks)} task ({len(algos)} thuật toán x "
        f"{len(dataset_names)} dataset x {num_runs} run). "
        f"Đã xong {len(done)}, còn lại {len(tasks)}."
    )

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            fh.flush()

        max_workers = 1 if args.smoke else os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_task, algo, ds, run_id): (algo, ds, run_id)
                for algo, ds, run_id in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="heldout tasks"):
                algo, ds, run_id = futures[future]
                try:
                    row = future.result()
                    writer.writerow(row)
                    fh.flush()
                except Exception as exc:  # noqa: BLE001
                    print(f"[LỖI] {algo} x {ds} x run{run_id}: {exc}")

    write_summary()


if __name__ == "__main__":
    main()
