
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
N_QUERY_REPS = 200                                                      
N_BATCH_REPS = 20                                         


def _load(name: str):
    df = pd.read_csv(os.path.join(PROC_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _time_knn(X: np.ndarray, y: np.ndarray, n_feat: int, seed: int = 42) -> dict:
    n_feat = max(1, min(n_feat, X.shape[1]))
    Xs = X[:, :n_feat]
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, stratify=y, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)


    clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, algorithm="brute").fit(X_tr, y_tr)

                                                                      
    clf.predict(X_te[:min(5, len(X_te))])
    clf.predict(X_te)

                                                                      
    q = X_te[:1]
    times = []
    for _ in range(N_QUERY_REPS):
        t0 = time.perf_counter()
        clf.predict(q)
        times.append(time.perf_counter() - t0)
    per_query_ms = float(np.median(times)) * 1000

                                                                      
    times_b = []
    for _ in range(N_BATCH_REPS):
        t0 = time.perf_counter()
        clf.predict(X_te)
        times_b.append(time.perf_counter() - t0)
    batch_ms = float(np.median(times_b)) * 1000

    return {"n_feat": n_feat, "per_query_ms": per_query_ms, "batch_ms": batch_ms,
            "n_test": len(X_te)}


def _time_synthetic(d: int, n_train: int, n_test: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    y_tr = rng.integers(0, 2, n_train)
    X_te = rng.standard_normal((n_test, d)).astype(np.float32)
    clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, algorithm="brute").fit(X_tr, y_tr)
    clf.predict(X_te[:5]); clf.predict(X_te)           
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
