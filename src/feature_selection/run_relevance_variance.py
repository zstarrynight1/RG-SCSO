"""Phân tích variance của relevance field ρ trên dataset gene-expression n
nhỏ (Q1 review Priority 6 / W9) — ColonCancer (n=62) và Leukemia (n=72) có
p >> n (2000/3571 feature), nên ρ_j = MI(X_j;y)/H(y) tính lại trên từng
train-fold có nguy cơ overfitting/bất ổn định cao nhưng bài chính CHỈ đánh
giá gián tiếp qua accuracy hạ nguồn, chưa đo trực tiếp độ ổn định của CHÍNH
relevance field.

PHƯƠNG PHÁP: bootstrap resampling (n_boot lần, mỗi lần lấy mẫu CÓ HOÀN LẠI
kích thước n từ chính dataset), tính lại ρ trên mỗi resample, rồi báo cáo:
    - Spearman correlation trung bình giữa các cặp ρ resample (ổn định thứ
      hạng feature qua các lần resample).
    - Jaccard overlap trung bình của tập "top-K feature liên quan nhất"
      (K = round(mean RG-SCSO subset size trên dataset đó, đọc từ
      fs_results.csv nếu có, fallback 50) giữa các cặp resample — đo trực
      tiếp độ ổn định của CHÍNH tập feature mà RMS sẽ ưu tiên giữ lại.
    - Std của ρ_j từng feature qua các resample (trung bình toàn bộ feature).

So sánh thêm với 3 dataset THẤP chiều/mẫu lớn hơn (Sonar n=208, WDBC n=569,
Zoo n=101) để có đường tham chiếu — nếu ColonCancer/Leukemia RÕ RỆT kém ổn
định hơn, đây là bằng chứng trực tiếp cho rủi ro overfitting relevance field
mà W9 nêu; nếu KHÔNG khác biệt đáng kể, rủi ro W9 nêu không được xác nhận
thực nghiệm trên dữ liệu này.

Output: experiments/results_relevance_variance/relevance_variance_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_relevance_variance
        [--smoke] [--datasets ...] [--n-boot N]
"""

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
    """Trả về ma trận (n_boot, n_features) — ρ tính lại trên mỗi resample có
    hoàn lại kích thước n. Resample nào chỉ còn 1 class (hiếm với n nhỏ, lớp
    thiểu số) được resample lại tới khi hợp lệ, tối đa 20 lần thử."""
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

    # Spearman trung bình giữa mọi cặp resample (thứ hạng feature ổn định?).
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
