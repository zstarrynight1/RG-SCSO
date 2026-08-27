"""R2 PILOT — kiểm định go/no-go cho RG-SCSO TRƯỚC khi mở full run.

QUAN TRỌNG (spec 4.2 + nguyên tắc "làm 1 lần duy nhất"): đây là PILOT, KHÔNG
phải số báo cáo trong paper. Mục đích duy nhất: quyết định có đủ tín hiệu để
chạy full experiment hay không. Số liệu chính thức của paper đến từ full run
`run_feature_selection.py` dưới protocol đã khóa (EXPERIMENT_PROTOCOL.md).

Thiết kế: 5 dataset (2 gene high-dim + 3 spread) × PILOT_RUNS run độc lập, so
RG-SCSO với SCSO (base) và AOA (đang dẫn FS). Dùng ĐÚNG config gốc (pop=30,
max_iter=500, seed=RANDOM_SEED_BASE+run_id, cùng fitness/CV) để pilot có tính
dự báo cho full run.

STOP-GATE ĐĂNG KÝ TRƯỚC: RG-SCSO phải vượt CẢ SCSO LẪN AOA (theo mean accuracy)
trên >= GATE_MIN / 5 dataset. Không đạt -> DỪNG, báo user, chỉnh THIẾT KẾ cơ
chế (không tinh chỉnh seed/param), rồi validate lại.

Chạy: python -m src.feature_selection.run_fs_pilot
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
from src.feature_selection.fitness import (
    evaluate_binary_mask,
    finalize_solution,
    make_fitness_function,
)

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_pilot")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "pilot_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
PILOT_DATASETS = ["Leukemia", "ColonCancer", "Sonar", "WDBC", "Zoo"]
PILOT_ALGORITHMS = ["RG-SCSO", "SCSO", "AOA"]
PILOT_RUNS = 10
GATE_MIN = 3  # RG-SCSO phải vượt cả SCSO lẫn AOA trên >= GATE_MIN/5 dataset


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _run_single(algorithm: str, dataset: str, run_id: int) -> dict:
    X, y = _load(dataset)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    if algorithm == "RG-SCSO":
        def eval_mask(mask: np.ndarray) -> float:
            return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

        result = RGSCSO(
            obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
            X=X, y=y, eval_mask=eval_mask,
        ).optimize()
        final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    elif algorithm == "SCSO":
        result = SCSO(
            obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed
        ).optimize()
        final = finalize_solution(result["best_solution"], X, y, seed=seed)
    else:
        result = run_mealpy_baseline(
            algorithm, obj_func, dim=dim, lb=SEARCH_LB, ub=SEARCH_UB,
            pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed,
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


def evaluate_gate(df: pd.DataFrame) -> bool:
    """In bảng mean accuracy và kiểm tra stop-gate. Trả về True nếu ĐẠT."""
    mean_acc = df.groupby(["dataset", "algorithm"])["accuracy"].mean().unstack()
    mean_nf = df.groupby(["dataset", "algorithm"])["n_selected_features"].mean().unstack()
    print("\n=== MEAN ACCURACY (pilot) ===")
    print(mean_acc.reindex(columns=PILOT_ALGORITHMS).round(4).to_string())
    print("\n=== MEAN #FEATURES ===")
    print(mean_nf.reindex(columns=PILOT_ALGORITHMS).round(1).to_string())

    wins = 0
    print("\n=== STOP-GATE (RG-SCSO vượt CẢ SCSO lẫn AOA?) ===")
    for ds in PILOT_DATASETS:
        rg = mean_acc.loc[ds, "RG-SCSO"]
        beats_scso = rg > mean_acc.loc[ds, "SCSO"]
        beats_aoa = rg > mean_acc.loc[ds, "AOA"]
        ok = bool(beats_scso and beats_aoa)
        wins += ok
        print(f"  {ds:14} RG={rg:.4f}  >SCSO={beats_scso}  >AOA={beats_aoa}  -> {'PASS' if ok else 'no'}")
    passed = wins >= GATE_MIN
    print(f"\nRG-SCSO vượt cả 2 baseline trên {wins}/{len(PILOT_DATASETS)} dataset "
          f"(cần >= {GATE_MIN}). => {'ĐẠT — mở full run' if passed else 'CHƯA ĐẠT — dừng, chỉnh thiết kế'}")
    return passed


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = [
        (a, ds, r)
        for a in PILOT_ALGORITHMS
        for ds in PILOT_DATASETS
        for r in range(PILOT_RUNS)
    ]
    print(f"Pilot: {len(PILOT_ALGORITHMS)} thuật toán x {len(PILOT_DATASETS)} dataset x "
          f"{PILOT_RUNS} run = {len(tasks)} task.")

    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_run_single, a, ds, r): (a, ds, r) for a, ds, r in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pilot"):
            a, ds, r = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[LỖI] {a} x {ds} x run{r}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nĐã ghi {RESULTS_CSV} ({len(df)} dòng).")
    evaluate_gate(df)


if __name__ == "__main__":
    main()
