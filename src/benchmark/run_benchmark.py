"""Phase 2 — chạy benchmark: ECL-SCSO + SCSO gốc + 9 baseline (mealpy) trên
bộ hàm CEC2017 (hoặc fallback cổ điển nếu opfunu không cài được), mỗi cặp
(algorithm, function) chạy NUM_INDEPENDENT_RUNS lần độc lập, seed =
RANDOM_SEED_BASE + run_id để reproducible.

LƯU Ý: spec gốc (PROJECT_SPEC.md mục 3.4 và 4.2) không khớp số lượng thuật
toán (mục 3.4 liệt kê 9 baseline, mục 4.2 nói "10 thuật toán = ECL-SCSO +
SCSO + 8 baseline"). Đã hỏi và được xác nhận dùng ĐỦ 9 baseline, tổng cộng
11 thuật toán (ECL-SCSO, SCSO, GA, PSO, GWO, WOA, HHO, SSA, AOA, COA, OOA).

Output:
    experiments/results_benchmark/benchmark_results.csv
        cột: algorithm, function_name, run_id, best_fitness, runtime_seconds
    experiments/results_benchmark/summary_stats.csv
        cột: algorithm, function_name, mean, std, best, worst, median

Chạy: python -m src.benchmark.run_benchmark
"""

from __future__ import annotations

import csv
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import (
    MAX_ITERATION,
    NUM_INDEPENDENT_RUNS,
    POPULATION_SIZE,
    RANDOM_SEED_BASE,
)
from src.algorithms.baselines import run_mealpy_baseline
from src.algorithms.ecl_scso import ECLSCSO
from src.algorithms.scso import SCSO
from src.benchmark.cec_functions import build_function, get_function_names

BENCHMARK_DIM = 30  # dimension chuẩn của bộ CEC2017
OUTPUT_DIR = os.path.join("experiments", "results_benchmark")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_stats.csv")

MEALPY_BASELINES = ["GA", "PSO", "GWO", "WOA", "HHO", "SSA", "AOA", "COA", "OOA"]
ALL_ALGORITHMS = ["SCSO", "ECL-SCSO"] + MEALPY_BASELINES


def _run_one_algorithm_on_function(algorithm: str, function_name: str, dim: int) -> list[dict]:
    """Chạy 1 thuật toán trên 1 hàm benchmark, NUM_INDEPENDENT_RUNS lần độc lập.

    Build lại BenchmarkFunction NGAY TRONG worker (thay vì truyền obj_func đã
    build sẵn qua process boundary) vì closure/object opfunu không pickle
    được an toàn qua ProcessPoolExecutor (multiprocessing trên macOS dùng
    'spawn').
    """
    bf = build_function(function_name, dim=dim)
    rows = []
    for run_id in range(NUM_INDEPENDENT_RUNS):
        seed = RANDOM_SEED_BASE + run_id
        if algorithm == "SCSO":
            result = SCSO(
                obj_func=bf.obj_func,
                dim=bf.dim,
                lb=bf.lb,
                ub=bf.ub,
                pop_size=POPULATION_SIZE,
                max_iter=MAX_ITERATION,
                seed=seed,
            ).optimize()
        elif algorithm == "ECL-SCSO":
            result = ECLSCSO(
                obj_func=bf.obj_func,
                dim=bf.dim,
                lb=bf.lb,
                ub=bf.ub,
                pop_size=POPULATION_SIZE,
                max_iter=MAX_ITERATION,
                seed=seed,
            ).optimize()
        else:
            result = run_mealpy_baseline(
                algorithm,
                bf.obj_func,
                dim=bf.dim,
                lb=bf.lb,
                ub=bf.ub,
                pop_size=POPULATION_SIZE,
                max_iter=MAX_ITERATION,
                seed=seed,
            )
        rows.append(
            {
                "algorithm": algorithm,
                "function_name": function_name,
                "run_id": run_id,
                "best_fitness": result["best_fitness"],
                "runtime_seconds": result["runtime"],
            }
        )
    return rows


def _write_results_csv(rows: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["algorithm", "function_name", "run_id", "best_fitness", "runtime_seconds"]
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(rows: list[dict]) -> None:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["algorithm"], row["function_name"])
        groups.setdefault(key, []).append(row["best_fitness"])

    fieldnames = ["algorithm", "function_name", "mean", "std", "best", "worst", "median"]
    with open(SUMMARY_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (algorithm, function_name), values in sorted(groups.items()):
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "function_name": function_name,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "best": min(values),
                    "worst": max(values),
                    "median": statistics.median(values),
                }
            )


def main() -> None:
    function_names = get_function_names(dim=BENCHMARK_DIM)
    tasks = [(algo, fname) for algo in ALL_ALGORITHMS for fname in function_names]

    print(
        f"Chạy {len(ALL_ALGORITHMS)} thuật toán x {len(function_names)} hàm benchmark "
        f"x {NUM_INDEPENDENT_RUNS} run = {len(tasks) * NUM_INDEPENDENT_RUNS} lần optimize "
        f"(dim={BENCHMARK_DIM}, pop={POPULATION_SIZE}, max_iter={MAX_ITERATION})."
    )

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(_run_one_algorithm_on_function, algo, fname, BENCHMARK_DIM): (
                algo,
                fname,
            )
            for algo, fname in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="benchmark tasks"):
            algo, fname = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 — muốn thấy rõ task nào lỗi
                print(f"[LỖI] {algo} x {fname}: {exc}")

    _write_results_csv(all_rows)
    _write_summary_csv(all_rows)
    print(f"Đã ghi {RESULTS_CSV} ({len(all_rows)} dòng) và {SUMMARY_CSV}.")


if __name__ == "__main__":
    main()
