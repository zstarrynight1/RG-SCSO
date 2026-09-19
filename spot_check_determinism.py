
from __future__ import annotations

import pandas as pd

from src.feature_selection.run_fs_ablation import RESULTS_CSV, _run_single


SPOT = [
    ("Full", "ColonCancer", 5),
    ("NoRMS", "ColonCancer", 5),
    ("Full", "Leukemia", 3),
    ("NoUMR", "Sonar", 0),
    ("NoORL", "WDBC", 0),
]


def main() -> None:
    saved = pd.read_csv(RESULTS_CSV)
    print(f"{'config':14s}{'dataset':13s}{'run':>4s}  "
          f"{'acc(CSV)':>10s}{'acc(re)':>10s}  {'nf(CSV)':>8s}{'nf(re)':>7s}  verdict")
    all_ok = True
    for cfg, ds, rid in SPOT:
        row = saved[(saved.config_name == cfg) & (saved.dataset == ds)
                    & (saved.run_id == rid)].iloc[0]
        re = _run_single(cfg, ds, rid)
        acc_ok = abs(row.accuracy - re["accuracy"]) < 1e-12
        nf_ok = int(row.n_selected_features) == int(re["n_selected_features"])
        ok = acc_ok and nf_ok
        all_ok &= ok
        print(f"{cfg:14s}{ds:13s}{rid:>4d}  "
              f"{row.accuracy:>10.6f}{re['accuracy']:>10.6f}  "
              f"{int(row.n_selected_features):>8d}{int(re['n_selected_features']):>7d}  "
              f"{'MATCH' if ok else 'MISMATCH <<<'}")
    print()
    print("KẾT LUẬN:", "TẤT CẢ TRÙNG KHÍT — throttle vô hại, dùng được kết quả."
          if all_ok else "CÓ LỆCH — phải re-run toàn bộ ablation.")


if __name__ == "__main__":
    main()
