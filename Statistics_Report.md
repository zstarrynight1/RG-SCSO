# Statistics Report — RG-SCSO (IJCS)

> **Mục đích (SOP 1 Q7 / F2):** nêu tường minh estimand, đơn vị thống kê, quy tắc pairing, test, hiệu chỉnh đa so sánh, effect size, CI, sample-size rationale, và phân biệt confirmatory/exploratory — cho **từng** phân tích.
> **Nguồn:** `src/stats/statistical_tests.py`, raw `experiments/results_fs_heldout/`, commit `246b142`.

## 1. Hai tầng phân tích (đơn vị thống kê KHÁC nhau — nêu rõ)

| # | Phân tích | Đơn vị thống kê | Vai trò |
|---|---|---|---|
| **A** | **Friedman + Nemenyi CD trên held-out** | **18 datasets** (mỗi dataset = 1 quan sát; rank trung bình qua datasets) | **CONFIRMATORY / PRIMARY** — bằng chứng ranking chính |
| B | Per-dataset Wilcoxon signed-rank | **30 runs / dataset** (paired theo seed) | Supporting — per-dataset significance |
| C | Bootstrap CI paired-difference | 18 dataset-level means | Supporting — quantify độ lớn advantage |

Implement (`statistical_tests.py`): (A) `friedman_test_and_ranking` pivot theo dataset, `ranks.mean(axis=0)` → unit = dataset; (B) `paired_wilcoxon_vs_target` group theo dataset, pivot theo run → `wilcoxon()` trên 30 run; Holm formed **per dataset**.

## 2. 10 mục SOP 1 Q7

1. **Estimand:** với mỗi cặp (RG-SCSO, baseline), sự khác biệt về **held-out accuracy trung bình ở cấp dataset**, tổng hợp qua 18 datasets (analysis A/C). Analysis B: khác biệt phân phối accuracy giữa 2 thuật toán trên **cùng 1 dataset** qua 30 run paired.
2. **Experimental unit:** một (dataset × algorithm × run) độc lập, sinh bởi outer 80/20 split + seed.
3. **Statistical unit:** **dataset** (A/C, primary) ; **run** (B, per-dataset).
4. **Pairing:** paired theo **seed** (analysis B, cùng seed→cùng split) và theo **dataset** (A/C, cùng dataset cho mọi thuật toán). Mọi so sánh là paired.
5. **Primary test:** Friedman omnibus + Nemenyi critical-difference (held-out, 7 algorithms × 18 datasets).
6. **Post-hoc / multiplicity:** Wilcoxon signed-rank pairwise + **Holm correction** (formed **per dataset** cho analysis B; α=0.05). Nemenyi tự kiểm soát đa so sánh cho analysis A.
7. **Effect size:** Cohen's d + rank-biserial r (`cohens_d`, `rank_biserial_r`).
8. **CI (bổ sung F2):** **paired-difference bootstrap 95% CI** (RG-SCSO − baseline) trên 18 dataset-level means, B=10,000, seed=42 → `experiments/results_fs_heldout/paired_diff_ci.csv`:

   | Baseline | Mean Δacc | 95% CI | Wins/18 |
   |---|---:|---|---:|
   | AOA | +0.0170 | [+0.0031, +0.0338] | 12/18 |
   | SCSO | +0.0545 | [+0.0304, +0.0839] | 18/18 |
   | RIME | +0.0614 | [+0.0341, +0.0952] | 18/18 |
   | PSO | +0.0657 | [+0.0386, +0.0988] | 18/18 |
   | COA | +0.0507 | [+0.0303, +0.0763] | 18/18 |
   | GWO | +0.0753 | [+0.0408, +0.1175] | 18/18 |

   → CI của **mọi** cặp loại trừ 0; kể cả đối thủ gần nhất (AOA) advantage là **dương và significant** ở cấp dataset, nhưng **độ lớn nhỏ (moderate)** — hỗ trợ hạ claim CR2 thành "moderate aggregate advantage".

9. **Sample-size rationale:** 30 runs/cell là quy ước repeated-stochastic-run trong benchmark metaheuristic (khớp SCSO/mealpy literature); **đây là repeated stochastic runs + dataset-level inference, KHÔNG phải independent trials**. 18 datasets là toàn bộ benchmark family (không phải sample từ population lớn hơn) → inference mang tính *benchmark-specific*, không general-population. Không có formal power analysis a-priori; giới hạn này được ghi nhận.
10. **Confirmatory vs exploratory:** **held-out (A) = confirmatory** (prior/search/CV chỉ trên train). **In-sample per-dataset (B) = exploratory / optimistic upper bound** — manuscript đã disclose "paired per-dataset comparisons, not independent trials" và effect size "contract to held-out". Enrichment/mechanism analyses = exploratory/correlational.

## 3. Pseudo-replication — xử lý minh bạch

Analysis B coi 30 run như đơn vị ⇒ là *within-dataset repeated-run* inference (không phải cross-dataset generalization). Để tránh over-interpret: **primary claim chỉ dựa trên analysis A (dataset-level Friedman/CD) + analysis C (paired-diff CI)**; analysis B chỉ mô tả per-dataset. Manuscript cần giữ đúng ranh giới này (đã phần lớn tuân thủ ở Results §3.3).

## 4. Còn thiếu / khuyến nghị

- (Optional) a-priori power/MDE nếu muốn nâng lên confirmatory mạnh hơn — hiện scope là benchmark methodology, chấp nhận được nếu ghi rõ.
- Đã bổ sung paired-difference CI (mục 8) — nên đưa 1 dòng vào Results/Discussion thay cho việc chỉ report CI của mean từng thuật toán.
