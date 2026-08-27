#!/usr/bin/env bash
# Driver: chạy family (resume) rồi robustness (KNN,SVM) — resume-safe, ghi log cố định.
# Chạy detached: setsid nohup ./run_experiments.sh >/dev/null 2>&1 < /dev/null &
set -u
cd /Users/hynee/Desktop/ecl-scso-feature-selection
source .venv/bin/activate 2>/dev/null
export PYTHONUNBUFFERED=1
LOG=experiments/run_experiments.log
echo "=== DRIVER START $(date) pid=$$ ===" >> "$LOG"
echo "=== FAMILY START $(date) ===" >> "$LOG"
python -m src.feature_selection.run_fs_scso_family >> "$LOG" 2>&1
echo "=== FAMILY DONE $(date) ===" >> "$LOG"
echo "=== ROBUSTNESS (KNN,SVM) START $(date) ===" >> "$LOG"
python -m src.feature_selection.run_fs_robustness --wrappers KNN,SVM >> "$LOG" 2>&1
echo "=== ALL DONE $(date) ===" >> "$LOG"
