
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
POLL_SECONDS = 600                                                         


def rime_datasets_done() -> int:
    try:
        raw = pd.read_csv(CSV, on_bad_lines="skip")
    except Exception:
        return -1
    cnt = raw[raw.algorithm == "RIME"].groupby("dataset").size()
    return sum(1 for d in ALL_DS if cnt.get(d, 0) >= 30)


def main() -> None:
    while rime_datasets_done() < 18:
        time.sleep(POLL_SECONDS)

                                                                 
    subprocess.run([sys.executable, "build_paper_structure.py"], check=True)
    subprocess.run([sys.executable, "build_paper_tex.py"], check=True)
    print("RIME 18/18 hoàn tất — đã regenerate RG-SCSO_IEEE_draft.docx + RG-SCSO_demo.tex")


if __name__ == "__main__":
    main()
