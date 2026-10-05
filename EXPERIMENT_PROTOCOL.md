# EXPERIMENT PROTOCOL — RG-SCSO (khóa trước khi chạy full)

> **Pre-registration.** Tài liệu này khóa thiết kế thí nghiệm TRƯỚC khi chạy full
> experiment và TRƯỚC khi xem kết quả chính thức. Không sửa tiêu chí sau khi đã
> thấy số. Mọi thay đổi (nếu bắt buộc) phải ghi log có ngày + lý do ở cuối file,
> không xóa nội dung cũ. Nguyên tắc tối cao: **làm 1 lần duy nhất, trung thực.**
> Ngày khóa: 2026-07-01.

## 1. Thuật toán đề xuất

**RG-SCSO** — Relevance-Guided Sand Cat Optimizer (binary-native feature selection).
Diễn giải lại "sensitivity range" của SCSO thành độ nhạy lật-bit theo từng feature,
dẫn hướng bởi trường độ-liên-quan ρ học online. Ba thành phần:

- **C1 — Relevance-Modulated Sensitivity (RMS):** `src/feature_selection/transfer_function.py::binarize_relevance`
- **C2 — Online Relevance Learning (ORL):** `src/feature_selection/relevance.py::RelevanceField`
- **C3 — Uncertainty-Targeted Memetic Refinement (UMR):** `src/algorithms/rg_scso.py::_memetic_refine`

Cài đặt: `src/algorithms/rg_scso.py::RGSCSO`.

## 2. Baseline (bộ rút gọn, khóa)

RG-SCSO vs **SCSO** (2022, base — bắt buộc so), **AOA** (2021, đang dẫn FS — bắt
buộc vượt), **COA/CoatiOA** (2023), **GWO** (2014), **PSO** (1995). Tùy chọn bổ
sung 1 phương pháp binary-FS 2022–2024 nếu tích hợp kịp. Bỏ nhóm yếu (GA, WOA,
HHO, SSA, OOA).

## 3. Dataset (khóa)

18 dataset UCI đã tiền xử lý trong `data/processed/`, gồm 2 gene-expression
high-dim làm case study: **Leukemia** (3571 feature, 72 mẫu), **ColonCancer**
(2000 feature, 62 mẫu). Không thêm/bớt dataset sau khi khóa.

## 4. Giao thức (khóa — công bằng tuyệt đối)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| pop_size | 30 | `config.POPULATION_SIZE` |
| max_iter | 500 | `config.MAX_ITERATION` |
| số run độc lập | 30 | `config.NUM_INDEPENDENT_RUNS` |
| seed | `RANDOM_SEED_BASE(42) + run_id` | dùng CHUNG mọi thuật toán (paired) |
| classifier fitness | KNN (k=5) | `config.KNN_NEIGHBORS` |
| CV | StratifiedKFold k=5, scaler fit trong từng fold | `fitness._knn_cv_accuracy` |
| fitness | `0.99·(1−acc) + 0.01·(n_sel/n_total)` | `config.FITNESS_ALPHA/BETA` |
| không gian liên tục | dim=n_features, [−1, 1] | `run_feature_selection.SEARCH_LB/UB` |

Khác biệt duy nhất giữa các thuật toán = bản thân thuật toán. Tham số RG-SCSO
(gamma=0.5, umr_k=8, ema_lambda=0.9, w_online=0.3, delta_scale=0.01) khóa ở
default trong `RGSCSO.__init__`; KHÔNG tinh chỉnh theo dataset để ra số đẹp.

## 5. Metric & kiểm định thống kê (khóa)

- Metric chính: **accuracy** (KNN 5-fold). Phụ: fitness, số feature được chọn.
- **Wilcoxon signed-rank** (paired theo seed) so RG-SCSO với từng baseline trên
  từng dataset; **hiệu chỉnh Holm** trong mỗi dataset.
- **Effect size bắt buộc**: Cohen's d (paired) + rank-biserial r (đi kèm Wilcoxon).
- **Friedman test** + average ranking toàn bộ dataset; **Nemenyi post-hoc +
  Critical-Difference diagram**.
- Tất cả ở `src/stats/statistical_tests.py`. Báo cáo TOÀN BỘ, kể cả dataset thua.

## 6. Tiêu chí thành công (KHÓA — quyết định "có đóng góp" hay không)

Điều kiện tối thiểu để tuyên bố RG-SCSO là đóng góp thật:

1. **Cải thiện Friedman rank so với SCSO gốc** (rank RG-SCSO < rank SCSO), VÀ
2. **Vượt AOA** trên đa số dataset theo mean accuracy, VÀ
3. Trên các cặp so sánh có ý nghĩa: **Wilcoxon + Holm p < 0.05** với **effect
   size ≥ small** (|d| ≥ 0.2 hoặc |r| ≥ 0.2) ở đa số.

Nếu KHÔNG đạt → báo cáo trung thực là null/negative, KHÔNG tinh chỉnh số; bàn
với user hướng chỉnh THIẾT KẾ cơ chế rồi validate lại.

## 7. Ablation (Falsifiability — khóa)

5 cấu hình Full/NoRMS/NoORL/NoUMR/NoImprovement (`run_fs_ablation.py`) × 5 dataset
đại diện × 30 run. Thành phần nào gỡ ra mà accuracy KHÔNG giảm có ý nghĩa thống
kê → KHÔNG load-bearing → **cắt bỏ** khỏi thuật toán cuối.

## 8. Chứng minh nhân-quả (nhấn mạnh cái mới)

Ngoài "thắng", đo trực tiếp cơ chế: (a) overlap giữa feature RG-SCSO chọn và
top-relevance ρ; (b) đường hội tụ so baseline; (c) số feature (parsimony);
(d) sự thay đổi ρ_online qua vòng lặp trên dataset gene high-dim.

## 9. Trình tự thực thi có stop-gate

R0 stats → R1 code+smoke (DONE) → **R2 pilot** (`run_fs_pilot.py`, 5×10, gate
≥3/5) → nếu ĐẠT → R3 full run (`run_feature_selection.py`, cần AC power +
caffeinate) → R4 stats+ablation+figures → R5 viết theo Q1_BLUEPRINT.md.

Pilot KHÔNG phải số báo cáo; số paper = full run duy nhất dưới protocol này.

---
### Change log
- 2026-07-01: khóa lần đầu (trước full run).
- 2026-07-01: **Cân bằng ngân sách NFE** (trước pilot, trước khi xem kết quả full).
  Phát hiện ở phép thử tín hiệu: RG-SCSO có memetic (C3) tốn thêm ~4000 eval/run
  so với baseline → không công bằng. Bổ sung tham số `max_nfe` (mặc định =
  pop_size × max_iter = 15000, KHỚP baseline SCSO/mealpy) + bộ đếm NFE trong
  `RGSCSO`; dừng khi hết ngân sách. Đây là siết chặt tính công bằng, KHÔNG phải
  tinh chỉnh để ra số đẹp (thực tế làm RG-SCSO KHÓ hơn, ~394 vòng thay vì 500).
  Mục 4 (giao thức) áp dụng ràng buộc NFE chung cho MỌI thuật toán.
- 2026-07-02: **Pilot R2 ĐẠT 5/5** (Leukemia/ColonCancer/Sonar/WDBC/Zoo, 10 run,
  budget-matched). RG-SCSO vượt CẢ SCSO lẫn AOA cả 5 dataset về accuracy VÀ chọn
  ~½ số feature. Đây KHÔNG phải số báo cáo; mở R3 full run theo protocol này.
- 2026-07-02: **Chốt baseline SOTA recent = RIME** (Rime Optimization Algorithm,
  Su et al., Neurocomputing 2023) — hiện thực hóa tùy chọn "+1 binary-FS SOTA
  2022–2024" đã pre-register ở Mục 2. Chạy CÙNG pipeline binarize như baseline
  khác (công bằng tuyệt đối). Bộ so sánh cuối = 7: RG-SCSO, SCSO, AOA, COA, GWO,
  PSO, RIME. Baseline cũ (SCSO/AOA/COA/GWO/PSO, 18×30) tái dùng từ fs_results.csv
  (code path chung không đổi → tái lập được, paired theo seed); chỉ RG-SCSO +
  RIME chạy mới.
- 2026-07-08: **R3b — Fold-honest held-out generalization run (khóa TRƯỚC khi chạy,
  chưa xem số full).** Động cơ: confound audit phát hiện `relevance_prior` tính MI
  trên TOÀN BỘ (X, y) gồm nhãn test fold → RG-SCSO có leak transductive nhẹ (mean
  Spearman ρ_full vs ρ_train = 0.75; nặng trên GermanCredit 0.32, SpectEW 0.34,
  gene-set top-k Jaccard 0.48–0.60). Đây là NHÁNH VALIDATION BỔ SUNG, KHÔNG thay
  main results in-sample R3 (in-sample là protocol wrapper-FS chuẩn, giữ nguyên +
  khai báo MI-on-train). Giao thức khóa: mỗi (algo,dataset,run) tách outer 80/20
  stratified (random_state=seed); MI prior + search + fitness (KNN 5-fold CV) CHỈ
  trên train-80; báo cáo accuracy trên held-out-20. Budget khớp (pop×iter=15000).
  Bộ 7 thuật toán + 18 dataset + 30 run y hệt main. Metric of record =
  heldout_accuracy; kiểm định Wilcoxon+Holm+Cohen's d + Friedman/CD như Mục 5.
  Tiêu chí thành công GIỮ NGUYÊN Mục 6 (không hạ chuẩn sau khi thấy pilot). Pilot
  R3b (8 ds, 3 run, budget-matched) ĐẠT 7 WIN/0/1 LOSS vs SCSO — KHÔNG phải số báo
  cáo, chỉ go/no-go. Code: `src/feature_selection/run_fs_heldout.py`. Báo cáo TRUNG
  THỰC dù held-out margin nhỏ hơn in-sample (chênh in-sample phần lớn do optimistic
  bias của CV dùng chung — công bằng cho ranking, thổi effect size tuyệt đối).
