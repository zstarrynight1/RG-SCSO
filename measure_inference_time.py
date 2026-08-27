"""Đo chi phí inference (wall-clock) như hàm của SỐ CHIỀU (§2.2 Diem_yeu_RG-SCSO.md:
"giá trị thực tiễn của parsimony là gì?"). KNN k-NN suy luận có chi phí O(n_train·d)
mỗi truy vấn (tính khoảng cách), nên đo trực tiếp latency dự đoán ở d = số feature
trung bình mỗi thuật toán chọn (RG-SCSO/SCSO/AOA, từ fs_results.csv đã có) trên
CÙNG dữ liệu thật, KHÔNG cần mask đã chọn thật sự (không lưu) — vì latency KNN phụ
thuộc SỐ chiều của phép tính khoảng cách, không phụ thuộc feature nào cụ thể. Đây
là phép đo CHI PHÍ TÍNH TOÁN của việc chọn subset nhỏ hơn, tách biệt khỏi accuracy
(đã báo cáo riêng ở bảng khác) — không đại diện cho pipeline suy luận đầy đủ.

Output: experiments/results_inference/inference_time.csv
Chạy:   .venv/bin/python measure_inference_time.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import KNN_NEIGHBORS

PROC_DIR = os.path.join("data", "processed")
FS_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
OUT_DIR = os.path.join("experiments", "results_inference")
OUT_CSV = os.path.join(OUT_DIR, "inference_time.csv")

DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
ALGOS = ["RG-SCSO", "SCSO", "AOA"]
N_QUERY_REPS = 200   # số lần lặp đo latency 1 truy vấn (median ổn định)
N_BATCH_REPS = 20    # số lần lặp đo latency cả batch test


def _load(name: str):
    df = pd.read_csv(os.path.join(PROC_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _time_knn(X: np.ndarray, y: np.ndarray, n_feat: int, seed: int = 42) -> dict:
    """Fit KNN(k=5) trên n_feat cột đầu (đại diện chi phí tính toán ở số chiều
    này, KHÔNG phải mask đã chọn thật — xem docstring module)."""
    n_feat = max(1, min(n_feat, X.shape[1]))
    Xs = X[:, :n_feat]
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, stratify=y, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    # algorithm="brute" CỐ ĐỊNH: sklearn "auto" tự chọn ball_tree/kd_tree khác
    # nhau tùy (n_samples, n_features) của TỪNG dataset, làm độ trễ không còn
    # phản ánh sạch chi phí O(n_train*d) của khoảng cách — đo lệch, gây nhiễu
    # không tương quan với d (phát hiện được khi kiểm tra sơ bộ). brute force
    # còn là lựa chọn ĐÚNG về mặt thực tế cho các tập chiều cao trong bài
    # (ColonCancer/Leukemia hàng nghìn feature) vì cây kd/ball suy biến về
    # brute-force do curse of dimensionality.
    clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, algorithm="brute").fit(X_tr, y_tr)

    # warm-up (tránh chi phí import/cache lần gọi đầu làm méo phép đo)
    clf.predict(X_te[:min(5, len(X_te))])
    clf.predict(X_te)

    # per-query latency (1 mẫu/lần) — mô phỏng suy luận thời gian thực
    q = X_te[:1]
    times = []
    for _ in range(N_QUERY_REPS):
        t0 = time.perf_counter()
        clf.predict(q)
        times.append(time.perf_counter() - t0)
    per_query_ms = float(np.median(times)) * 1000

    # batch latency (toàn bộ test set/lần) — mô phỏng suy luận theo lô
    times_b = []
    for _ in range(N_BATCH_REPS):
        t0 = time.perf_counter()
        clf.predict(X_te)
        times_b.append(time.perf_counter() - t0)
    batch_ms = float(np.median(times_b)) * 1000

    return {"n_feat": n_feat, "per_query_ms": per_query_ms, "batch_ms": batch_ms,
            "n_test": len(X_te)}


def _time_synthetic(d: int, n_train: int, n_test: int, seed: int) -> float:
    """Đo latency KNN(k=5, brute) trên dữ liệu TỔNG HỢP ở quy mô lớn (deployment-
    scale), số chiều d = số feature thật RG-SCSO/AOA chọn trên dataset tương ứng.
    Tách biệt khỏi phần đo trên dataset thật (quá nhỏ để tín hiệu vượt nhiễu) —
    đây là minh họa CÓ KIỂM SOÁT cho lập luận độ phức tạp O(n_train·d), không
    phải số đo trên chính 18 dataset benchmark."""
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    y_tr = rng.integers(0, 2, n_train)
    X_te = rng.standard_normal((n_test, d)).astype(np.float32)
    clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, algorithm="brute").fit(X_tr, y_tr)
    clf.predict(X_te[:5]); clf.predict(X_te)  # warm-up
    times = []
    for _ in range(N_BATCH_REPS):
        t0 = time.perf_counter()
        clf.predict(X_te)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = pd.read_csv(FS_CSV)

    print("=== (1) Đo trực tiếp trên 18-dataset benchmark (n_train nhỏ: 50-455) ===")
    rows = []
    for ds in DATASETS:
        X, y = _load(ds)
        for a in ALGOS:
            n_feat = round(fs[(fs.algorithm == a) & (fs.dataset == ds)]
                            .n_selected_features.mean())
            r = _time_knn(X, y, n_feat)
            r.update({"dataset": ds, "algorithm": a})
            rows.append(r)
            print(f"{ds:12s} {a:8s} n_feat={r['n_feat']:5d}  "
                  f"per_query={r['per_query_ms']:.4f} ms  batch={r['batch_ms']:.3f} ms")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nĐã ghi {OUT_CSV} ({len(out)} dòng).")
    piv = out.pivot_table(index="dataset", columns="algorithm", values="batch_ms")
    print("\nRG-SCSO vs AOA batch-latency delta (ms) per dataset (n_train quá nhỏ "
          "→ kỳ vọng nhiễu, không phải tín hiệu sạch):")
    print((piv["AOA"] - piv["RG-SCSO"]).round(3))

    print("\n=== (2) Mô phỏng kiểm soát quy mô lớn (n_train=5000, n_test=2000, "
          "d = số feature THẬT RG-SCSO/AOA chọn) ===")
    rows2 = []
    for ds in DATASETS:
        d_rg = round(fs[(fs.algorithm == "RG-SCSO") & (fs.dataset == ds)]
                      .n_selected_features.mean())
        d_aoa = round(fs[(fs.algorithm == "AOA") & (fs.dataset == ds)]
                       .n_selected_features.mean())
        t_rg = _time_synthetic(d_rg, 5000, 2000, seed=42)
        t_aoa = _time_synthetic(d_aoa, 5000, 2000, seed=42)
        speedup = (t_aoa - t_rg) / t_aoa * 100
        rows2.append({"dataset": ds, "d_rgscso": d_rg, "d_aoa": d_aoa,
                       "batch_ms_rgscso": t_rg, "batch_ms_aoa": t_aoa,
                       "speedup_pct": speedup})
        print(f"{ds:12s} d_RG={d_rg:5d} ({t_rg:7.2f} ms)  d_AOA={d_aoa:5d} "
              f"({t_aoa:7.2f} ms)  speedup={speedup:+.1f}%")
    out2 = pd.DataFrame(rows2)
    out2.to_csv(os.path.join(OUT_DIR, "inference_time_synthetic.csv"), index=False)
    print(f"\nmean synthetic speedup: {out2['speedup_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
