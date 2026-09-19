
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import RANDOM_SEED_BASE
from src.feature_selection.relevance import relevance_prior

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_relevance_variance")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "relevance_variance_results.csv")

DEFAULT_DATASETS = ["ColonCancer", "Leukemia", "Sonar", "WDBC", "Zoo"]
DEFAULT_N_BOOT = 30
DEFAULT_TOP_K = 50


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _bootstrap_rhos(X: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    rhos = []
    for b in range(n_boot):
        for _attempt in range(20):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) >= 2:
                break
        rho = relevance_prior(X[idx], y[idx], seed=seed + b, method="mi")
        rhos.append(rho)
    return np.array(rhos)


def _analyze(dataset: str, n_boot: int, top_k: int, seed: int) -> dict:
    X, y = _load(dataset)
    d = X.shape[1]
    k = min(top_k, d)
    rhos = _bootstrap_rhos(X, y, n_boot, seed)

                                                                            
    n_pairs = 0
    spearman_sum = 0.0
    jaccard_sum = 0.0
    top_sets = [set(np.argsort(-rhos[b])[:k]) for b in range(n_boot)]
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            rho_corr, _ = spearmanr(rhos[i], rhos[j])
            spearman_sum += 0.0 if np.isnan(rho_corr) else rho_corr
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccard_sum += inter / union if union else 0.0
            n_pairs += 1

    return {
        "dataset": dataset, "n_samples": X.shape[0], "n_features": d,
        "n_boot": n_boot, "top_k": k,
        "mean_spearman_between_resamples": spearman_sum / n_pairs,
        "mean_jaccard_topk_between_resamples": jaccard_sum / n_pairs,
        "mean_std_rho_per_feature": float(np.mean(np.std(rhos, axis=0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="Zoo, n_boot=3 — kiểm tra wiring, không phải kết quả.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, n_boot = ["Zoo"], 3
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        n_boot = args.n_boot

    rows = [_analyze(ds, n_boot, args.top_k, RANDOM_SEED_BASE) for ds in datasets]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.smoke:
        print("\nSMOKE OK — wiring chạy đúng. KHÔNG dùng làm kết quả.")
        return

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nĐã ghi {RESULTS_CSV}")


if __name__ == "__main__":
    main()
