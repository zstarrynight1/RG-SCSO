"""Phân tích ablation: mỗi removal (NoRMS/NoORL/NoUMR) vs Full, paired theo seed
(run_id) trên từng dataset. Wilcoxon signed-rank + Holm + Cohen's d.

Quy tắc (Falsifiability, spec 8.1/4.2): thành phần LOAD-BEARING nếu gỡ ra làm
accuracy GIẢM CÓ Ý NGHĨA (p_holm<0.05, delta>0) trên >=1 dataset. Nếu gỡ ra
KHÔNG bao giờ giảm significant -> KHÔNG load-bearing -> CẮT (không giữ trang trí).

Output: experiments/results_fs/fs_ablation_summary.csv
Chạy: .venv/bin/python analyze_fs_ablation.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.stats.statistical_tests import cohens_d, holm_correction

CSV = os.path.join("experiments", "results_fs", "fs_ablation_results.csv")
OUT = os.path.join("experiments", "results_fs", "fs_ablation_summary.csv")
REMOVALS = ["NoRMS", "NoORL", "NoUMR"]
COMPONENT = {"NoRMS": "C1 (RMS)", "NoORL": "C2 (ORL)", "NoUMR": "C3 (UMR)"}
ALPHA = 0.05


def main() -> None:
    df = pd.read_csv(CSV)
    datasets = sorted(df.dataset.unique())

    rows = []
    for removal in REMOVALS:
        for ds in datasets:
            full = (df[(df.config_name == "Full") & (df.dataset == ds)]
                    .sort_values("run_id")["accuracy"].to_numpy())
            rem = (df[(df.config_name == removal) & (df.dataset == ds)]
                   .sort_values("run_id")["accuracy"].to_numpy())
            delta = full - rem  # >0 => gỡ thành phần làm TỆ đi
            if np.allclose(delta, 0.0):
                p = 1.0  # mọi run bằng nhau (vd Leukemia bão hòa)
            else:
                p = wilcoxon(full, rem, zero_method="wilcox").pvalue
            rows.append(dict(
                component=COMPONENT[removal], removal=removal, dataset=ds,
                full_mean=full.mean(), removed_mean=rem.mean(),
                delta=delta.mean(), cohens_d=cohens_d(full, rem), p_value=p))

    res = pd.DataFrame(rows)
    res["p_holm"] = holm_correction(res["p_value"].to_numpy())
    res["degrades_sig"] = (res["p_holm"] < ALPHA) & (res["delta"] > 0)
    res["improves_sig"] = (res["p_holm"] < ALPHA) & (res["delta"] < 0)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print(res[["component", "dataset", "full_mean", "removed_mean", "delta",
               "cohens_d", "p_value", "p_holm", "degrades_sig"]]
          .round(4).to_string(index=False))

    print("\n=== VERDICT theo component ===")
    for removal in REMOVALS:
        sub = res[res.removal == removal]
        deg = sub[sub.degrades_sig]
        imp = sub[sub.improves_sig]
        verdict = "GIỮ (load-bearing)" if len(deg) else "CẮT (không load-bearing)"
        where = ", ".join(f"{r.dataset}(Δ{r.delta:+.4f})" for r in deg.itertuples())
        print(f"{COMPONENT[removal]:10s}: gỡ ra giảm significant trên "
              f"{len(deg)}/{len(sub)} dataset -> {verdict}"
              + (f"  [{where}]" if where else "")
              + (f"  | CẢNH BÁO: gỡ ra lại TĂNG significant trên "
                 f"{len(imp)} dataset" if len(imp) else ""))


if __name__ == "__main__":
    main()
