"""Hyperparameter sensitivity study cho RG-SCSO (trả lời reviewer #4).

Phân tích one-factor-at-a-time (OFAT): mỗi siêu tham số được quét quanh giá trị
mặc định trong khi các tham số còn lại giữ nguyên default, để chứng minh RG-SCSO
KHÔNG được tinh chỉnh may rủi / cherry-pick (spec 8.1/4.2 — báo cáo trung thực).

QUAN TRỌNG — bản cuối chỉ có HAI siêu tham số thật:
    gamma (γ) : cường độ điều biến relevance ở RMS (C1)   — bản cuối (ORL off)
    umr_k (K) : số feature "biên" thử lật mỗi vòng ở UMR (C3) — bản cuối (ORL off)
λ (ema_lambda) và w_o (w_online) CHỈ tác động lên ORL — thành phần đã bị ablation
CẮT (use_orl=False trong bản ship). Với bản cuối chúng TRƠ (ρ = ρ_static). Ta vẫn
quét chúng trên biến thể ORL-ON để chứng minh: (i) không nhạy cảm, (ii) ORL-on
không thắng ORL-off — củng cố quyết định cắt. KHÔNG quét tham số trơ trên bản cuối.

Ngân sách NFE giữ CỐ ĐỊNH (max_nfe = pop×iter) cho mọi cấu hình → so sánh công
bằng, không cấu hình nào được "nhiều eval hơn".

Output: experiments/results_fs_sensitivity/fs_sensitivity_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_sensitivity [--smoke]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.rg_scso import RGSCSO
from src.feature_selection.fitness import evaluate_binary_mask, make_fitness_function

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_sensitivity")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_sensitivity_results.csv")

SEARCH_LB, SEARCH_UB = -1.0, 1.0
N_RUNS = 10  # số run độc lập cho robustness check (không phải headline stat test)

# datasets trải chiều: thấp (Zoo d=16) / trung (Sonar d=60) / cao gene (ColonCancer d=2000)
DATASETS = ["Zoo", "Sonar", "ColonCancer"]

# default của bản cuối (ORL OFF = bản ship) và của biến thể ORL-ON
BASE_FINAL = dict(use_rms=True, use_orl=False, use_umr=True,
                  gamma=0.5, umr_k=8, ema_lambda=0.9, w_online=0.3)
BASE_ORL = dict(use_rms=True, use_orl=True, use_umr=True,
                gamma=0.5, umr_k=8, ema_lambda=0.9, w_online=0.3)

# OFAT: (param, [values], base_config, nhãn config)
SWEEPS = [
    ("gamma", [0.0, 0.25, 0.5, 0.75, 1.0], BASE_FINAL, "final"),   # γ — bản cuối
    ("umr_k", [2, 4, 8, 12, 16], BASE_FINAL, "final"),             # K — bản cuối
    ("ema_lambda", [0.7, 0.9, 0.99], BASE_ORL, "orl"),            # λ — biến thể ORL
    ("w_online", [0.1, 0.3, 0.7], BASE_ORL, "orl"),               # w_o — biến thể ORL
]


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _build_tasks(n_runs: int) -> list[dict]:
    """Sinh danh sách task; dedup các cấu hình trùng (điểm default xuất hiện ở
    nhiều sweep) để không lãng phí compute — mỗi cấu hình duy nhất chạy 1 lần."""
    seen: dict[tuple, dict] = {}
    for param, values, base, tag in SWEEPS:
        for v in values:
            cfg = dict(base)
            cfg[param] = v
            key = (cfg["use_orl"], cfg["gamma"], cfg["umr_k"],
                   cfg["ema_lambda"], cfg["w_online"])
            entry = seen.setdefault(key, {"cfg": cfg, "sweeps": []})
            entry["sweeps"].append((param, v, tag))
    tasks = []
    for key, entry in seen.items():
        for ds in DATASETS:
            for r in range(n_runs):
                tasks.append({"key": key, "cfg": entry["cfg"],
                              "sweeps": entry["sweeps"], "dataset": ds, "run_id": r})
    return tasks


def _run_single(task: dict) -> dict:
    cfg, ds, run_id = task["cfg"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    dim = X.shape[1]
    seed = RANDOM_SEED_BASE + run_id
    obj_func = make_fitness_function(X, y, seed=seed)

    def eval_mask(mask: np.ndarray) -> float:
        return evaluate_binary_mask(mask, X, y, seed=seed)["fitness"]

    opt_kwargs = {k: cfg[k] for k in
                  ("use_rms", "use_orl", "use_umr", "gamma", "umr_k", "ema_lambda", "w_online")}
    result = RGSCSO(
        obj_func, dim, SEARCH_LB, SEARCH_UB, POPULATION_SIZE, MAX_ITERATION, seed,
        X=X, y=y, eval_mask=eval_mask, **opt_kwargs,
    ).optimize()
    final = evaluate_binary_mask(result["best_mask"], X, y, seed=seed)
    # một dòng cho MỖI sweep mà cấu hình này thuộc về (default point dùng chung)
    rows = []
    for param, value, tag in task["sweeps"]:
        rows.append({
            "sweep": param, "value": value, "config_tag": tag, "dataset": ds,
            "run_id": run_id, "fitness": final["fitness"], "accuracy": final["accuracy"],
            "n_selected_features": final["n_selected_features"],
            "runtime_seconds": result["runtime"],
        })
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="chạy nhẹ: 1 run, chỉ Zoo, để kiểm tra wiring (không phải kết quả).")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.smoke:
        global DATASETS
        DATASETS = ["Zoo"]
        tasks = _build_tasks(1)
    else:
        tasks = _build_tasks(N_RUNS)

    n_cfg = len({t["key"] for t in tasks})
    print(f"Sensitivity: {n_cfg} cấu hình duy nhất x {len(DATASETS)} dataset x "
          f"{1 if args.smoke else N_RUNS} run = {len(tasks)} task.")

    rows = []
    workers = 1 if args.smoke else os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sensitivity"):
            try:
                rows.extend(fut.result()["rows"])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['cfg']} x {t['dataset']} x run{t['run_id']}: {exc}")

    df = pd.DataFrame(rows)
    if args.smoke:
        print(df.to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Đã ghi {RESULTS_CSV} ({len(df)} dòng).")
    for param, values, base, tag in SWEEPS:
        sub = df[df.sweep == param]
        print(f"\n=== {param}  (config={tag}) — mean accuracy theo dataset ===")
        print(sub.groupby(["dataset", "value"])["accuracy"].mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
