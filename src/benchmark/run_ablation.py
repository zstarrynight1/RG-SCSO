"""Phase 4 — Ablation study (PROJECT_SPEC.md mục 6.2): chạy lại ECL-SCSO với
6 cấu hình (bật/tắt độc lập 4 cải tiến) trên 6 hàm CEC2017 đại diện (KHÔNG
chạy lại hết 29 hàm để tiết kiệm thời gian, đúng như spec cho phép), mỗi cấu
hình x mỗi hàm chạy NUM_INDEPENDENT_RUNS lần độc lập.

6 hàm đại diện (phủ đủ 4 nhóm của CEC2017): F1 (unimodal), F4 (simple
multimodal — Rastrigin), F9 (simple multimodal — Schwefel), F13 (hybrid),
F20 (composition), F27 (composition).

6 cấu hình:
    Full              : chaotic=on,  adaptiveR=on,  DE=on,  Levy=on
    OnlyChaoticInit   : chaotic=on,  adaptiveR=off, DE=off, Levy=off
    OnlyAdaptiveR     : chaotic=off, adaptiveR=on,  DE=off, Levy=off
    OnlyDEMutation    : chaotic=off, adaptiveR=off, DE=on,  Levy=off
    OnlyLevyFlight    : chaotic=off, adaptiveR=off, DE=off, Levy=on
    NoImprovement     : chaotic=off, adaptiveR=off, DE=off, Levy=off  (= SCSO gốc)

Output:
    experiments/results_benchmark/ablation_results.csv
        cột: config_name, function_name, run_id, best_fitness, runtime_seconds

Chạy: python -m src.benchmark.run_ablation
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
from src.algorithms.ecl_scso import ECLSCSO
from src.benchmark.cec_functions import build_function

BENCHMARK_DIM = 30
OUTPUT_DIR = os.path.join("experiments", "results_benchmark")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "ablation_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_summary.csv")

ABLATION_FUNCTIONS = [
    "CEC2017_F1",
    "CEC2017_F4",
    "CEC2017_F9",
    "CEC2017_F13",
    "CEC2017_F20",
    "CEC2017_F27",
]

ABLATION_CONFIGS = {
    "Full": dict(
        use_chaotic_init=True, use_adaptive_R=True, use_de_mutation=True, use_levy_flight=True
    ),
    "OnlyChaoticInit": dict(
        use_chaotic_init=True, use_adaptive_R=False, use_de_mutation=False, use_levy_flight=False
    ),
    "OnlyAdaptiveR": dict(
        use_chaotic_init=False, use_adaptive_R=True, use_de_mutation=False, use_levy_flight=False
    ),
    "OnlyDEMutation": dict(
        use_chaotic_init=False, use_adaptive_R=False, use_de_mutation=True, use_levy_flight=False
    ),
    "OnlyLevyFlight": dict(
        use_chaotic_init=False, use_adaptive_R=False, use_de_mutation=False, use_levy_flight=True
    ),
    "NoImprovement": dict(
        use_chaotic_init=False, use_adaptive_R=False, use_de_mutation=False, use_levy_flight=False
    ),
}


def _run_config_on_function(config_name: str, function_name: str, dim: int) -> list[dict]:
    bf = build_function(function_name, dim=dim)
    flags = ABLATION_CONFIGS[config_name]
    rows = []
    for run_id in range(NUM_INDEPENDENT_RUNS):
        seed = RANDOM_SEED_BASE + run_id
        result = ECLSCSO(
            obj_func=bf.obj_func,
            dim=bf.dim,
            lb=bf.lb,
            ub=bf.ub,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATION,
            seed=seed,
            **flags,
        ).optimize()
        rows.append(
            {
                "config_name": config_name,
                "function_name": function_name,
                "run_id": run_id,
                "best_fitness": result["best_fitness"],
                "runtime_seconds": result["runtime"],
            }
        )
    return rows


def _write_results_csv(rows: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["config_name", "function_name", "run_id", "best_fitness", "runtime_seconds"]
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(rows: list[dict]) -> None:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["config_name"], row["function_name"])
        groups.setdefault(key, []).append(row["best_fitness"])

    fieldnames = ["config_name", "function_name", "mean", "std", "best", "worst", "median"]
    with open(SUMMARY_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (config_name, function_name), values in sorted(groups.items()):
            writer.writerow(
                {
                    "config_name": config_name,
                    "function_name": function_name,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "best": min(values),
                    "worst": max(values),
                    "median": statistics.median(values),
                }
            )


def main() -> None:
    tasks = [(cfg, fname) for cfg in ABLATION_CONFIGS for fname in ABLATION_FUNCTIONS]
    print(
        f"Ablation: {len(ABLATION_CONFIGS)} cấu hình x {len(ABLATION_FUNCTIONS)} hàm x "
        f"{NUM_INDEPENDENT_RUNS} run = {len(tasks) * NUM_INDEPENDENT_RUNS} lần optimize."
    )

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(_run_config_on_function, cfg, fname, BENCHMARK_DIM): (cfg, fname)
            for cfg, fname in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="ablation tasks"):
            cfg, fname = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[LỖI] {cfg} x {fname}: {exc}")

    _write_results_csv(all_rows)
    _write_summary_csv(all_rows)
    print(f"Đã ghi {RESULTS_CSV} ({len(all_rows)} dòng) và {SUMMARY_CSV}.")


if __name__ == "__main__":
    main()
