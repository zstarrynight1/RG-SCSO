"""Phase 3 — chạy feature selection: ECL-SCSO + SCSO gốc + 9 baseline (mealpy)
trên 18 dataset UCI đã chuẩn hóa (`data/processed/*.csv`), mỗi cặp
(algorithm, dataset) chạy NUM_INDEPENDENT_RUNS lần độc lập, seed =
RANDOM_SEED_BASE + run_id để reproducible.

GHI CHÚ THIẾT KẾ QUAN TRỌNG:
    - Mỗi đánh giá fitness (KNN + 5-fold CV) tốn vài-trăm ms đến ~90ms tùy
      dataset — ĐẮT hơn nhiều so với hàm benchmark toán học ở Phase 2. Ước
      tính tổng thời gian cho toàn bộ grid (11 thuật toán x 18 dataset x 30
      run, pop=30, max_iter=500 đúng config.py) là ~475 giờ CPU (~59 giờ
      wall-clock với 8 worker song song, ~2.5 ngày). Đã xác nhận với người
      dùng chạy đúng config gốc trong nền nhiều ngày.
    - Vì 1 dataset (WaveformEW, n=5000) có thể tốn ~23 phút CHO MỖI run, nếu
      gộp 30 run vào 1 task multiprocessing thì 1 task có thể chiếm 1 worker
      suốt ~11.6 giờ, làm mất cân bằng tải. Do đó task được chia ở granularity
      (algorithm, dataset, run_id) — mỗi task chỉ chạy 1 seed — để 8 worker
      được tận dụng đều hơn và tiến độ ghi CSV theo từng dòng (resilient nếu
      bị ngắt giữa chừng, không mất hết kết quả).
    - Search space liên tục cho mỗi dataset: dim = n_features, lb=-1, ub=1
      (convention phổ biến trong literature binary-wrapper-FS qua sigmoid
      transfer function, ví dụ binary GWO của Emary et al.) — KHÔNG có trong
      spec gốc nên ghi rõ ở đây làm tài liệu đối chiếu.

Output (ghi tăng dần — append từng dòng ngay khi 1 task xong, KHÔNG đợi toàn
bộ job hoàn tất, để không mất dữ liệu nếu job bị ngắt giữa chừng):
    experiments/results_fs/fs_results.csv
        cột: algorithm, dataset, run_id, fitness, accuracy,
             n_selected_features, n_total_features, runtime_seconds
    experiments/results_fs/fs_summary.csv (tính lại từ fs_results.csv sau khi
        toàn bộ job xong, hoặc chạy riêng `--summary-only` để tính từ kết quả
        hiện có bất kỳ lúc nào trong lúc job đang chạy)

Chạy: python -m src.feature_selection.run_feature_selection
      python -m src.feature_selection.run_feature_selection --summary-only
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
from tqdm import tqdm

from config import (
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.rg_scso import RGSCSO
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import (
    evaluate_binary_mask,
    finalize_solution,
    make_fitness_function,
)

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "fs_summary.csv")

SEARCH_LB = -1.0
SEARCH_UB = 1.0

# Bộ thuật toán RÚT GỌN (Phase 3 redesign): giữ base SCSO + các baseline mạnh/gần
# đây, bỏ nhóm yếu thua nhất quán (GA, WOA, HHO, SSA, OOA). Kết quả cũ của
# SCSO/PSO/GWO/AOA/COA trên 18 dataset trong fs_results.csv TÁI DÙNG được (cùng
# protocol) — runner tự bỏ qua task đã xong, nên chỉ RG-SCSO cần chạy mới.
MEALPY_BASELINES = ["PSO", "GWO", "AOA", "COA", "RIME"]
ALL_ALGORITHMS = ["RG-SCSO", "SCSO"] + MEALPY_BASELINES

FIELDNAMES = [
    "algorithm",
    "dataset",
    "run_id",
    "fitness",
    "accuracy",
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


def _run_single_task(algorithm: str, dataset: str, run_id: int) -> dict:
    X, y = _load_dataset(dataset)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    if algorithm == "RG-SCSO":
        # Binary-native: chấm trực tiếp trên mask, chốt kết quả từ best_mask
        # (KHÔNG qua binarize_threshold của vị trí liên tục như baseline).
        # RG-SCSO cuối = 2 thành phần (RMS + UMR): dùng default use_orl=False của
        # RGSCSO — C2/ORL đã CẮT sau ablation R4 (không load-bearing). Số main run
        # vì thế KHỚP đúng thuật toán mô tả trong bài (bulletproof cho reviewer).
        def eval_mask(mask: np.ndarray) -> float:
            return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

        result = RGSCSO(
            obj_func=obj_func,
            dim=dim,
            lb=SEARCH_LB,
            ub=SEARCH_UB,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATION,
            seed=seed,
            X=X,
            y=y,
            eval_mask=eval_mask,
        ).optimize()
        final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
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
        final = finalize_solution(result["best_solution"], X, y, seed=seed)
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
        final = finalize_solution(result["best_solution"], X, y, seed=seed)
    return {
        "algorithm": algorithm,
        "dataset": dataset,
        "run_id": run_id,
        "fitness": final["fitness"],
        "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "runtime_seconds": result["runtime"],
    }


def _existing_completed_keys() -> set[tuple[str, str, int]]:
    """Đọc fs_results.csv hiện có (nếu job được chạy lại sau khi bị ngắt) để
    bỏ qua các task đã hoàn thành, không chạy lại từ đầu."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return {(row.algorithm, row.dataset, int(row.run_id)) for row in df.itertuples()}


def write_summary() -> None:
    df = pd.read_csv(RESULTS_CSV)
    rows = []
    for (algorithm, dataset), group in df.groupby(["algorithm", "dataset"]):
        for metric in ["accuracy", "fitness", "n_selected_features"]:
            values = group[metric].tolist()
            rows.append(
                {
                    "algorithm": algorithm,
                    "dataset": dataset,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "best": max(values) if metric != "fitness" else min(values),
                    "worst": min(values) if metric != "fitness" else max(values),
                }
            )
    pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric"]).to_csv(
        SUMMARY_CSV, index=False
    )
    print(f"Đã ghi {SUMMARY_CSV} ({len(rows)} dòng).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.summary_only:
        write_summary()
        return

    dataset_names = get_dataset_names()
    all_tasks = [
        (algo, ds, run_id)
        for algo in ALL_ALGORITHMS
        for ds in dataset_names
        for run_id in range(NUM_INDEPENDENT_RUNS)
    ]
    done = _existing_completed_keys()
    tasks = [t for t in all_tasks if t not in done]

    print(
        f"Tổng {len(all_tasks)} task ({len(ALL_ALGORITHMS)} thuật toán x "
        f"{len(dataset_names)} dataset x {NUM_INDEPENDENT_RUNS} run). "
        f"Đã xong {len(done)}, còn lại {len(tasks)}."
    )

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            fh.flush()

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(_run_single_task, algo, ds, run_id): (algo, ds, run_id)
                for algo, ds, run_id in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="fs tasks"):
                algo, ds, run_id = futures[future]
                try:
                    row = future.result()
                    writer.writerow(row)
                    fh.flush()
                except Exception as exc:  # noqa: BLE001 — muốn thấy rõ task nào lỗi
                    print(f"[LỖI] {algo} x {ds} x run{run_id}: {exc}")

    write_summary()


if __name__ == "__main__":
    main()
