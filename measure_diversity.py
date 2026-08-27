"""Đo đa dạng quần thể + tỉ lệ bit đóng băng theo vòng lặp (phòng thủ tử huyệt
§1.1 cân bằng E&E, §1.2 đóng băng bit trong Diem_yeu_RG-SCSO.md).

So RG-SCSO ở γ ∈ {0.0, 0.5, 1.0}: γ=0 là V-shaped thuần (không relevance), γ=0.5
là mặc định, γ=1.0 là bias cực đại. Nếu γ cao KHÔNG làm sụp đa dạng / đóng băng
bit sớm hơn rõ rệt so với γ=0 → cơ chế RMS an toàn về exploration (bác bỏ tử huyệt).
Ngược lại → phát hiện trung thực, cần sàn p_min.

Output: experiments/results_diversity/diversity_history.csv (trung bình theo run).
Chạy:   .venv/bin/python measure_diversity.py [--smoke]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.rg_scso import RGSCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROC_DIR = os.path.join("data", "processed")
OUT_DIR = os.path.join("experiments", "results_diversity")
OUT_CSV = os.path.join(OUT_DIR, "diversity_history.csv")
SEARCH_LB, SEARCH_UB = -1.0, 1.0

DATASETS = ["Zoo", "WDBC", "ColonCancer"]   # thấp / trung / siêu cao chiều
GAMMAS = [0.0, 0.5, 1.0]
N_RUNS = 3


def _load(name: str):
    df = pd.read_csv(os.path.join(PROC_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _one_run(X, y, gamma, seed):
    dim = X.shape[1]
    obj = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask):
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    opt = RGSCSO(obj, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
                 X=X, y=y, eval_mask=eval_mask, gamma=gamma, record_history=True)
    opt.optimize()
    return pd.DataFrame(opt.history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Zoo, γ=0.5, 1 run.")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = ["Zoo"] if args.smoke else DATASETS
    gammas = [0.5] if args.smoke else GAMMAS
    n_runs = 1 if args.smoke else N_RUNS

    rows = []
    for ds in datasets:
        X, y = _load(ds)
        for g in gammas:
            hists = [_one_run(X, y, g, RANDOM_SEED_BASE + r) for r in range(n_runs)]
            avg = pd.concat(hists).groupby("iter").mean().reset_index()
            avg["dataset"] = ds
            avg["gamma"] = g
            rows.append(avg)
            last = avg.iloc[-1]
            print(f"{ds:12s} γ={g:.1f}: end diversity={last['diversity']:.4f} "
                  f"frozen={last['frozen_frac']:.3f} size={last['mean_subset_size']:.1f}")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nĐã ghi {OUT_CSV} ({len(out)} dòng).")


if __name__ == "__main__":
    main()
