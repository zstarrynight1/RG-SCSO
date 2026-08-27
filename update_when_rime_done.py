"""Watcher: chờ RIME chạy xong đủ 18 dataset (>=30 run mỗi cái) trong
fs_results.csv, rồi regenerate CẢ docx lẫn tex một lần duy nhất và thoát.

Chạy nền:  .venv/bin/python update_when_rime_done.py
Số liệu vẫn auto-generate từ CSV (không gõ tay, không bịa). Watcher chỉ đợi
điều kiện đủ dữ liệu rồi gọi 2 script build có sẵn.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import pandas as pd

CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
ALL_DS = sorted(
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob("data/processed/*.csv")
)
POLL_SECONDS = 600  # 10 phút/lần — dữ liệu đổi theo giờ, không cần dày hơn


def rime_datasets_done() -> int:
    """Số dataset RIME đã đủ >=30 run. -1 nếu CSV đang bị ghi dở (đọc lỗi)."""
    try:
        raw = pd.read_csv(CSV, on_bad_lines="skip")
    except Exception:
        return -1
    cnt = raw[raw.algorithm == "RIME"].groupby("dataset").size()
    return sum(1 for d in ALL_DS if cnt.get(d, 0) >= 30)


def main() -> None:
    while rime_datasets_done() < 18:
        time.sleep(POLL_SECONDS)

    # Đủ 18/18 — regenerate cả hai deliverable từ cùng nguồn CSV.
    subprocess.run([sys.executable, "build_paper_structure.py"], check=True)
    subprocess.run([sys.executable, "build_paper_tex.py"], check=True)
    print("RIME 18/18 hoàn tất — đã regenerate RG-SCSO_IEEE_draft.docx + RG-SCSO_demo.tex")


if __name__ == "__main__":
    main()
