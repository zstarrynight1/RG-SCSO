"""Ablation study cho RG-SCSO trên feature selection — chứng minh mỗi thành
phần (C1/C2/C3) LOAD-BEARING (Falsifiability Test của Q1_BLUEPRINT).

5 cấu hình (bật/tắt độc lập 3 thành phần quanh cấu hình Full):
    Full          : RMS on,  ORL on,  UMR on
    NoRMS         : RMS off (V-shaped thuần, không relevance) — C1
    NoORL         : ORL off (prior tĩnh thuần, không học online) — C2
    NoUMR         : UMR off (không memetic refinement) — C3
    NoImprovement : tất cả off (≈ binary SCSO V-shaped trần)

Mỗi cấu hình × mỗi dataset × NUM_INDEPENDENT_RUNS run. Nếu gỡ 1 thành phần mà
accuracy KHÔNG giảm có ý nghĩa thống kê -> thành phần đó KHÔNG load-bearing ->
theo nguyên tắc nghiêm khắc, CẮT BỎ khỏi thuật toán cuối (không giữ trang trí).

Output: experiments/results_fs/fs_ablation_results.csv
Chạy: python -m src.feature_selection.run_fs_ablation
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, NUM_INDEPENDENT_RUNS, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.rg_scso import RGSCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_ablation_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0

# Dataset đại diện (spread độ chiều): 2 gene high-dim + 3 low/mid.
ABLATION_DATASETS = ["Leukemia", "ColonCancer", "Sonar", "WDBC", "Zoo"]

ABLATION_CONFIGS = {
    "Full": dict(use_rms=True, use_orl=True, use_umr=True),
    "NoRMS": dict(use_rms=False, use_orl=True, use_umr=True),
    "NoORL": dict(use_rms=True, use_orl=False, use_umr=True),
    "NoUMR": dict(use_rms=True, use_orl=True, use_umr=False),
    "NoImprovement": dict(use_rms=False, use_orl=False, use_umr=False),
}


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _run_single(config_name: str, dataset: str, run_id: int) -> dict:
    X, y = _load(dataset)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    result = RGSCSO(
        obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
        X=X, y=y, eval_mask=eval_mask, **ABLATION_CONFIGS[config_name],
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    return {
        "config_name": config_name,
        "dataset": dataset,
        "run_id": run_id,
        "fitness": final["fitness"],
        "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "runtime_seconds": result["runtime"],
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = [
        (cfg, ds, r)
        for cfg in ABLATION_CONFIGS
        for ds in ABLATION_DATASETS
        for r in range(NUM_INDEPENDENT_RUNS)
    ]
    print(f"FS ablation: {len(ABLATION_CONFIGS)} cấu hình x {len(ABLATION_DATASETS)} dataset x "
          f"{NUM_INDEPENDENT_RUNS} run = {len(tasks)} task.")

    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_run_single, c, ds, r): (c, ds, r) for c, ds, r in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fs ablation"):
            c, ds, r = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[LỖI] {c} x {ds} x run{r}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Đã ghi {RESULTS_CSV} ({len(df)} dòng).")
    print("\n=== MEAN ACCURACY theo cấu hình ===")
    print(df.groupby(["dataset", "config_name"])["accuracy"].mean().unstack()
          .reindex(columns=list(ABLATION_CONFIGS)).round(4).to_string())


if __name__ == "__main__":
    main()
