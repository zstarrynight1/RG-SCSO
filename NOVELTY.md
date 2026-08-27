# RG-SCSO — Cái mới & Tự đánh giá điều kiện Q1

> Bản nháp làm rõ đóng góp, soi theo Q1_BLUEPRINT Module −1/0. Viết TRONG lúc
> full run đang chạy — các con số dưới đây là dữ liệu full-run ĐÃ hoàn tất phần
> RG-SCSO + baseline (RIME đang chạy), CHƯA qua kiểm định thống kê. Không phải
> bản final; cập nhật sau khi có Wilcoxon/Holm/effect size + ablation.
> Ngày: 2026-07-03.

---

## 1. Cái mới cốt lõi (CORE_NOVELTY)

**Một câu:** RG-SCSO diễn giải lại "sensitivity range R" của SCSO — vốn là
tham số điều khiển explore→exploit trong *không gian liên tục* — thành **độ nhạy
lật-bit theo TỪNG feature** trong *không gian nhị phân*, được điều biến bởi một
**trường độ-liên-quan ρⱼ học online**.

**Vì sao đây là điểm mấu chốt (không phải "thêm 1 binary optimizer nữa"):**

Chẩn đoán từ Phase 3 (null result của ECL-SCSO) là gốc rễ của cái mới. Các cải
tiến continuous (chaotic init, adaptive R, DE, Lévy) **không truyền được** sang
kết quả nhị phân vì bước `sigmoid + ngưỡng 0.5` **lượng tử hóa mất** mọi tinh
chỉnh của toán tử liên tục — ta gọi là **"washout"**. Bằng chứng L1: Wilcoxon
0 win / 1 loss / 17 tie so với SCSO; Friedman ECL-SCSO (4.22) *dưới* cả SCSO gốc
(3.78). Đây là **cơ chế thất bại đo được**, không phải "hiệu năng kém" chung chung.

RG-SCSO chữa ĐÚNG bệnh đó: thay vì bơm tinh chỉnh vào không gian liên tục rồi
để nó bị nghiền nát ở khâu binarize, ta đưa quyết định lật-bit thành **native
binary** và gắn trực tiếp cơ chế R của SCSO vào xác suất lật, có tri thức bài
toán (ρ) dẫn hướng. Feature liên quan → khó bị bỏ; feature nhiễu → khó được thêm.

---

## 2. Phân loại novelty + Gap Architecture (Module 0)

**Novelty type: Type-B mạnh, nghiêng Type-A.**
- Type-B: tích hợp có-chứng-minh giữa (i) sensitivity range R của SCSO, (ii)
  transfer V-shaped (Mirjalili & Lewis 2013), (iii) trường relevance học online
  (credit-assignment kiểu EMA), (iv) memetic local search.
- Nghiêng Type-A ở chỗ: **việc diễn giải lại R thành độ nhạy per-feature là cơ
  chế chưa có tiền lệ cho SCSO** trong FS — không phải chỉ ghép transfer generic
  như các SCSO-FS 2022–2024.

**Gap Architecture (4 tầng — điền sẵn Module 0 §2):**
- **L1 — Empirical:** binary-wrapper-FS dùng transfer generic (sigmoid) mất
  thông tin tinh chỉnh của toán tử; số liệu null Phase 3 định lượng điều này.
- **L2 — Mechanistic:** thiếu cơ chế binary-native nối *search dynamics* (R) với
  *relevance bài toán* — đây là mechanism vắng gây ra L1, không phải "kém".
- **L3 — Literature:** paper 2022–2024 áp SCSO cho FS đều dùng transfer cố định,
  chưa ai diễn giải lại R theo hướng per-feature + relevance học online.
- **L4 — Consequence:** FS trên dữ liệu gene-expression chiều rất cao (hàng nghìn
  feature, vài chục mẫu) — nơi washout gây hại nặng nhất — bị ảnh hưởng trực tiếp.

---

## 3. Ba thành phần + Falsifiability Test

| | Thành phần | Cơ chế | Gỡ ra thì sao |
|---|---|---|---|
| **C1** | Relevance-Modulated Sensitivity (RMS) | V(x)=\|tanh(x)\| cho xác suất lật, điều biến theo ρ về "preferred bit" | → quay lại V-shaped thuần → mất dẫn hướng relevance |
| **C2** | Online Relevance Learning (ORL) | ρ = prior filter (MI/entropy) + EMA credit-assignment từ Δfitness | → chỉ còn prior tĩnh, mất thích nghi tương tác feature |
| **C3** | Uncertainty-Targeted Memetic Refinement (UMR) | greedy lật K feature có ρ gần 0.5 trên nghiệm best | → mất exploit precision ở chiều cao |

⚠ **Falsifiability CHƯA chứng minh xong.** Blueprint yêu cầu: gỡ mỗi component
mà accuracy KHÔNG giảm có ý nghĩa thống kê ⇒ component đó KHÔNG load-bearing ⇒
**PHẢI CẮT**. Ablation (5 config × 5 dataset × 30 run) chạy ở R4 sẽ quyết định
số component cuối. **Có thể C3 (hoặc C2) bị cắt** — nếu vậy ta báo trung thực và
thuật toán cuối gọn hơn. Đây là rủi ro thật, chưa khóa.

---

## 4. Bằng chứng hiện có (full-run, chưa kiểm định)

RG-SCSO đã xong 18/18 dataset × 30 run. So mean accuracy:

- **vs SCSO (base 2022): thắng 18/18**
- **vs AOA (đang dẫn FS 2021): thắng 18/18**
- Đứng **#1/18 dataset** trong nhóm 6 thuật toán đã có
- Parsimony: chọn ~½ số feature so với SCSO/AOA (vd Leukemia 931 vs 1782/3450;
  ColonCancer 558 vs 985/1988) — thắng KÉP accuracy + số feature
- Đã cân bằng **NFE = 15000 cho mọi thuật toán** (memetic C3 không được ngân sách
  eval nhiều hơn) → công bằng, chặn phản biện "thắng nhờ chạy nhiều hơn"

**CHƯA có (đang/ sẽ làm ở R4):** Wilcoxon+Holm p-value, effect size (Cohen's d /
rank-biserial r), Friedman rank + Nemenyi/CD diagram, ablation, bằng chứng
nhân-quả (overlap feature-chọn vs top-ρ), đường hội tụ. RIME (baseline SOTA 2023)
chưa vào bảng.

---

## 5. Đối chiếu Q1 red-flag checklist (Blueprint Module 0 §4)

| Check | Trạng thái | Ghi chú |
|---|---|---|
| **Baseline recency** (≤4 năm hoặc canonical) | ✅ ĐẠT | RIME 2023, COA 2023, SCSO 2022, AOA 2021 (recent); GWO 2014 + PSO 1995 gắn nhãn *canonical* |
| **Effect size** (Cohen's d / rank-biserial — BẮT BUỘC) | 🟡 SẼ CÓ | Code đã sẵn ở `statistical_tests.py`; chạy ở R4 |
| **Convergence analysis** | 🟡 SẼ CÓ | convergence_curve đã lưu mỗi run; vẽ ở R4 |
| **≥3 real-world validations** | ✅ ĐẠT (vượt) | 18 dataset UCI + 2 case study gene high-dim (Leukemia, ColonCancer) |
| **Wilcoxon + Friedman** (bắt buộc) | 🟡 SẼ CÓ | pipeline sẵn; chạy ở R4 |

Không có red-flag nào ở trạng thái ❌. 2 mục ✅, 3 mục chỉ chờ tính toán R4.

---

## 6. Điều kiện CÒN THIẾU để chắc suất Q1 (thành thật)

1. **Ablation phải chứng minh mỗi component load-bearing.** Đây là rủi ro lớn
   nhất. Nếu C2/C3 không load-bearing → cắt → đóng góp gọn lại (vẫn đăng được
   nhưng narrative đổi). CHƯA khóa.
2. **Significance + effect size phải giữ.** Thắng 18/18 mean accuracy KHÔNG đồng
   nghĩa 18/18 *significant*. Vài dataset (WDBC 0.980 vs 0.965) thắng sát → có
   thể p≥0.05. Ta báo cáo cả chỗ không significant.
3. **Bằng chứng nhân-quả, không chỉ "thắng".** Phải cho thấy ρ-guided ⇒ chọn
   feature liên quan hơn ⇒ mask nhỏ + acc cao (đo overlap chọn vs top-ρ). Thiếu
   cái này reviewer coi là "black-box thắng may".
4. **Kiểm tra phản xạ "too good to be true".** 18/18 sweep dễ khiến reviewer nghi.
   Phòng thủ: NFE-matched (đã làm), protocol pre-registered công khai, seed cố
   định, tái lập được, và RIME (SOTA 2023) vào bảng để có đối thủ mạnh gần đây.

---

## 7. Verdict (tự đánh giá thẳng)

**Đủ tiềm năng Q1 — có điều kiện.** Điểm mạnh hiếm có: (a) cái mới bám gốc sinh
học SCSO + chữa đúng bệnh washout đã đo được, không phải ghép tùy tiện; (b) tín
hiệu thực nghiệm rất mạnh và nhất quán (sweep + parsimony); (c) narrative gene
high-dim đúng gu SEC/KBS/ESWA.

**Nhưng chưa được tuyên bố "chắc Q1" cho tới khi:** ablation chốt component thật +
significance/effect size giữ + có bằng chứng nhân-quả. Nếu cả ba qua → đây là bài
Type-B mạnh, đủ sức nộp Swarm & Evolutionary Computation / Knowledge-Based Systems
/ Expert Systems with Applications. Nếu ablation cắt bớt component → vẫn đăng được
nhưng phải hạ tông đóng góp cho đúng sự thật.

**Nguyên tắc giữ nguyên:** báo cáo trung thực dù thắng hay thua, không chỉnh
param/seed cho đẹp số. Bản này cập nhật sau khi R4 xong.
