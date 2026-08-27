#!/usr/bin/env bash
# SVM wrapper trên ĐỦ 18 dataset (rev #3 — robustness classifier đầy đủ).
# Resume-safe: bỏ qua 5 dataset SVM đã chạy. Detached: nohup ./run_svm18.sh &
set -u
cd /Users/hynee/Desktop/ecl-scso-feature-selection
source .venv/bin/activate 2>/dev/null
export PYTHONUNBUFFERED=1
# 16 dataset khả thi cho SVM (bỏ KrVsKpEW n=3196 và WaveformEW n=5000: RBF-SVM
# O(n^2)/fit trong wrapper 15000-eval → nhiều tuần compute, bất khả thi).
DS="BreastEW,ColonCancer,Diabetes,GermanCredit,HeartDisease,IonosphereEW,Leukemia,Lymphography,M-of-n,Parkinsons,Sonar,SpectEW,TicTacToe,Vote,WDBC,Zoo"
echo "=== SVM/18 START $(date) ===" >> experiments/run_svm18.log
python -m src.feature_selection.run_fs_robustness --wrappers SVM --datasets "$DS" >> experiments/run_svm18.log 2>&1
echo "=== SVM/18 DONE $(date) ===" >> experiments/run_svm18.log
