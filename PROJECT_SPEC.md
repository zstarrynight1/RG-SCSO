# PROJECT SPEC: ECL-SCSO — Hybrid Metaheuristic for Feature Selection
## 0. BỐI CẢNH & MỤC TIÊU
Xây dựng một thuật toán metaheuristic mới gọi là **ECL-SCSO** (Enhanced Chaotic-Levy Sand Cat Swarm Optimization), là bản cải tiến của thuật toán gốc
**SCSO (Sand Cat Swarm Optimization, Seyyedabbasi & Kiani, 2022)**, sau đó:
1. Kiểm chứng ECL-SCSO trên benchmark function chuẩn (CEC2017 hoặc hàm benchmark cổ điển nếu CEC2017 không có sẵn thư viện Python tốt).
2. Áp dụng ECL-SCSO vào bài toán **Binary Feature Selection** trên 18 dataset UCI (chủ yếu y tế).
3. So sánh với 9 thuật toán baseline khác.
4. Chạy kiểm định thống kê (Wilcoxon, Friedman) và xuất toàn bộ bảng biểu, đồ thị cần cho một bài báo khoa học Q1.

Mục tiêu cuối: một thư mục project hoàn chỉnh, có thể chạy lại (reproducible),
xuất ra các file `.csv` (bảng số liệu) và `.png`/`.pdf` (đồ thị) sẵn sàng đưa
vào bài báo.
**Môi trường chạy:** MacBook Pro M2 Pro, CPU-only (không CUDA). Toàn bộ code
PHẢI chạy được thuần CPU, không phụ thuộc GPU.
## 1. CẤU TRÚC PROJECT (yêu cầu Claude Code tạo đúng cấu trúc này)
ecl-scso-feature-selection/
├── README.md
├── requirements.txt
├── config.py                      # toàn bộ hằng số/tham số tập trung ở đây
├── data/
│   ├── raw/                       # dataset gốc tải về
│   └── processed/                 # dataset đã chuẩn hóa (.csv chuẩn: cột cuối = label)
├── src/
│   ├── algorithms/
│   │   ├── base_optimizer.py      # class trừu tượng chung cho mọi optimizer
│   │   ├── scso.py                # SCSO gốc (chưa cải tiến) — bản baseline
│   │   ├── ecl_scso.py             # thuật toán đề xuất (4 cải tiến)
│   │   └── baselines.py           # wrapper gọi GA, PSO, GWO, WOA, HHO, SSA, AOA, COA, OOA (qua mealpy)
│   ├── benchmark/
│   │   ├── cec_functions.py       # định nghĩa các hàm benchmark
│   │   └── run_benchmark.py       # script chạy thí nghiệm benchmark
│   ├── feature_selection/
│   │   ├── fitness.py             # hàm fitness (KNN + k-fold CV)
│   │   ├── transfer_function.py   # sigmoid binary encoding
│   │   └── run_feature_selection.py
│   ├── stats/
│   │   └── statistical_tests.py   # Wilcoxon, Friedman, ranking
│   └── visualization/
│       └── plots.py               # convergence curve, boxplot, heatmap ranking
├── experiments/
│   ├── results_benchmark/         # output csv/json thí nghiệm benchmark
│   ├── results_fs/                # output csv/json thí nghiệm feature selection
│   └── figures/                   # toàn bộ hình xuất ra
├── notebooks/                     # (tùy chọn) notebook phân tích nhanh
└── tests/
    └── test_algorithms.py         # unit test cơ bản cho từng optimizer
```
## 2. THAM SỐ CHUNG (đặt trong `config.py`, KHÔNG hardcode rải rác)

```python
POPULATION_SIZE = 30
MAX_ITERATION = 500
NUM_INDEPENDENT_RUNS = 30          # mỗi thuật toán chạy 30 lần độc lập
RANDOM_SEED_BASE = 42              # seed = RANDOM_SEED_BASE + run_index, để reproducible
KFOLD = 5                          # k-fold cross-validation cho fitness
KNN_NEIGHBORS = 5
FITNESS_ALPHA = 0.99               # trọng số error rate
FITNESS_BETA = 0.01                # trọng số tỉ lệ feature được chọn
DIM_BINARY_THRESHOLD = 0.5         # ngưỡng sigmoid để binarize
```

**Yêu cầu bắt buộc:** mọi script chạy thí nghiệm phải set seed theo
`RANDOM_SEED_BASE + run_index` để đảm bảo có thể chạy lại đúng kết quả.

---

## 3. PHASE 1 — CODE THUẬT TOÁN (làm trước, test kỹ trước khi sang Phase 2)

### 3.1. `base_optimizer.py`
Tạo class trừu tượng `BaseOptimizer` với:
- `__init__(self, obj_func, dim, lb, ub, pop_size, max_iter, seed)`
- `optimize(self) -> dict` trả về `{"best_solution": ..., "best_fitness": ...,
  "convergence_curve": [...], "runtime": ...}`
- Mọi thuật toán (SCSO, ECL-SCSO, các baseline tự code) đều kế thừa class này
  để pipeline gọi đồng nhất.

### 3.2. `scso.py` — SCSO gốc
Implement đúng theo bài báo gốc Seyyedabbasi & Kiani (2022):
- Sensitivity range `R` giảm tuyến tính từ giá trị đầu (thường `S_M = 2`)
  về 0 theo vòng lặp.
- Pha exploration (random search quanh con mồi) và exploitation (spiral
  attack) theo đúng công thức gốc.
- **Yêu cầu:** viết docstring trích rõ công thức toán học (dạng comment),
  để sau này đối chiếu khi viết phần Methodology của bài báo.

### 3.3. `ecl_scso.py` — Thuật toán đề xuất (PHẦN QUAN TRỌNG NHẤT)

Implement **4 cải tiến**, mỗi cải tiến phải có thể tắt/mở độc lập qua flag
(để sau dùng cho ablation study ở Phase 4):

```python
class ECLSCSO(BaseOptimizer):
    def __init__(self, ..., use_chaotic_init=True, use_adaptive_R=True,
                 use_de_mutation=True, use_levy_flight=True,
                 stagnation_threshold=10, de_mutation_ratio=0.3, de_F=0.5):
        ...
```

**Cải tiến 1 — Chaotic Initialization (Tent map):**
```
x(n+1) = x(n)/0.7                  nếu x(n) < 0.7
x(n+1) = (10/3) * x(n) * (1-x(n))  nếu x(n) >= 0.7
```
Dùng Tent map để sinh giá trị khởi tạo quần thể thay vì `np.random.uniform`
thuần túy, sau đó scale về khoảng `[lb, ub]`.

**Cải tiến 2 — Adaptive Sensitivity Range (phi tuyến, thay tuyến tính gốc):**
```
R(t) = R_max * cos(pi/2 * t / max_iter)
```

**Cải tiến 3 — DE Mutation cho cá thể yếu:**
- Mỗi vòng lặp, sort quần thể theo fitness, lấy `de_mutation_ratio` (mặc định
  30%) cá thể tệ nhất.
- Với mỗi cá thể trong nhóm này, áp dụng:
```
V_i = X_r1 + de_F * (X_r2 - X_r3)
```
  với `r1, r2, r3` là index ngẫu nhiên khác nhau, khác `i`, lấy từ quần thể.
- Thêm bước crossover đơn giản (binomial crossover, crossover rate `CR=0.7`)
  giữa `V_i` và cá thể gốc `X_i`.
- Nếu fitness của cá thể mới tốt hơn → thay thế.

**Cải tiến 4 — Levy Flight khi trì trệ:**
- Theo dõi biến `stagnation_counter`: tăng 1 mỗi vòng lặp best fitness KHÔNG
  cải thiện; reset về 0 khi có cải thiện.
- Nếu `stagnation_counter >= stagnation_threshold`: áp dụng Levy flight cho
  best solution hiện tại để sinh 1 solution mới thử nghiệm:
```
step = Levy(lambda=1.5)   # dùng công thức Mantegna's algorithm
X_new = X_best + step * (X_best - X_random_other)
```
  Nếu `X_new` tốt hơn best hiện tại → cập nhật, reset stagnation_counter.

**Yêu cầu output bắt buộc:** convergence curve phải lưu best fitness ở MỖI
vòng lặp (không chỉ lưu giá trị cuối) để vẽ biểu đồ hội tụ.

### 3.4. `baselines.py`
Dùng thư viện `mealpy` (`pip install mealpy`) để lấy nhanh các baseline:
GA, PSO, GWO, WOA, HHO, SSA, AOA, COA (Coati), OOA (Osprey).
Viết wrapper để mọi baseline trả về cùng format dict như `BaseOptimizer`
(để pipeline downstream xử lý đồng nhất, không cần biết baseline nào).

> Nếu `mealpy` không có sẵn 1-2 thuật toán nào (ví dụ COA hoặc OOA quá mới),
> thông báo cho tôi biết, đừng tự ý thay bằng thuật toán khác mà không hỏi.

### 3.5. Unit test (`tests/test_algorithms.py`)
- Test ECL-SCSO chạy được trên hàm Sphere function (`f(x) = sum(x_i^2)`),
  dim=10, kiểm tra best_fitness phải tiến gần 0 sau 500 vòng lặp (sai số
  cho phép < 1e-3).
- Test convergence_curve có đúng độ dài = max_iter.
- Test với mỗi flag cải tiến tắt/mở riêng lẻ đều không bị lỗi runtime.

**ĐIỂM DỪNG PHASE 1:** Sau khi code xong, chạy thử ECL-SCSO vs SCSO gốc trên
2-3 hàm benchmark đơn giản (Sphere, Rastrigin, Ackley), in ra so sánh
best_fitness. Báo cáo lại cho tôi kết quả trước khi sang Phase 2.

---

## 4. PHASE 2 — BENCHMARK CEC

### 4.1. `cec_functions.py`
- Ưu tiên dùng thư viện có sẵn nếu tồn tại bản Python ổn định cho CEC2017
  hoặc CEC2022 (kiểm tra PyPI: `cec2017`, `opfunu`). Thư viện **`opfunu`**
  (PyPI) hỗ trợ tốt CEC2014/2017/2020/2022 bằng Python thuần — ưu tiên dùng
  cái này.
- Nếu không cài được, fallback dùng 10 hàm benchmark cổ điển (Sphere,
  Rastrigin, Ackley, Griewank, Rosenbrock, Schwefel, Zakharov, Michalewicz,
  Levy, Dixon-Price) — đủ dùng cho 1 bài Q1 tầm trung nếu CEC quá phức tạp.

### 4.2. `run_benchmark.py`
- Chạy **10 thuật toán** (ECL-SCSO + SCSO gốc + 8 baseline) × **bộ hàm
  benchmark đã chọn** × **30 lần độc lập**.
- Output: file `experiments/results_benchmark/benchmark_results.csv` với
  cột: `algorithm, function_name, run_id, best_fitness, runtime_seconds`.
- Output thứ 2: `experiments/results_benchmark/summary_stats.csv` với
  cột: `algorithm, function_name, mean, std, best, worst, median`.

**ĐIỂM DỪNG PHASE 2:** Báo cáo bảng `summary_stats.csv`, đặc biệt so sánh
ECL-SCSO vs SCSO gốc — phải thấy ECL-SCSO tốt hơn rõ ràng trên đa số hàm.
Nếu KHÔNG tốt hơn, dừng lại và báo tôi để xem lại cải tiến (không tự ý
"chỉnh" thuật toán cho ra kết quả đẹp).

---

## 5. PHASE 3 — FEATURE SELECTION PIPELINE

### 5.1. Dataset
Tải và xử lý 18 dataset UCI theo danh sách dưới đây. Chuẩn hóa mỗi dataset
thành file `.csv` trong `data/processed/`, format: các cột feature trước,
cột cuối cùng tên `label`.

| Dataset | Nguồn UCI / Kaggle | Ghi chú |
|---|---|---|
| BreastEW | UCI Breast Cancer Wisconsin (Diagnostic) | |
| WDBC | UCI WDBC | |
| SpectEW | UCI SPECT Heart | |
| Heart Disease (Cleveland) | UCI Heart Disease | |
| Parkinsons | UCI Parkinsons | |
| Diabetes (Pima Indians) | UCI/Kaggle Pima Indians Diabetes | |
| Lymphography | UCI Lymphography | |
| Colon Cancer | Kent Ridge Biomedical / Kaggle microarray | dim cao (~2000) |
| Leukemia | Kent Ridge Biomedical / Kaggle microarray | dim cao (~7129) |
| IonosphereEW | UCI Ionosphere | |
| Sonar | UCI Sonar | |
| Vote | UCI Congressional Voting Records | |
| Zoo | UCI Zoo | |
| M-of-n | tìm trong bộ "feature selection EW datasets" trên GitHub | |
| Tic-tac-toe | UCI Tic-Tac-Toe Endgame | |
| KrVsKpEW | UCI Chess (King-Rook vs King-Pawn) | |
| WaveformEW | UCI Waveform Database Generator | |
| German Credit | UCI Statlog German Credit Data | |

> Nếu một dataset không tìm được nguồn rõ ràng hoặc link UCI đã đổi, báo lại
> cho tôi thay vì tự thay bằng dataset khác.

### 5.2. `transfer_function.py`
```
S(x) = 1 / (1 + exp(-x))
binary_feature = 1 if random() < S(x) else 0
```

### 5.3. `fitness.py`
```
fitness = ALPHA * error_rate + BETA * (n_selected_features / n_total_features)
```
- `error_rate` tính bằng `1 - accuracy` của KNN (k=5) với k-fold
  cross-validation (k=5), dùng `scikit-learn`.
- Nếu một solution chọn 0 feature (toàn bộ binary = 0), gán fitness = 1
  (giá trị xấu nhất) để tránh lỗi chia 0 hoặc lựa chọn vô nghĩa.

### 5.4. `run_feature_selection.py`
- Chạy 10 thuật toán × 18 dataset × 30 lần độc lập.
- Output: `experiments/results_fs/fs_results.csv` với cột:
  `algorithm, dataset, run_id, fitness, accuracy, n_selected_features,
  n_total_features, runtime_seconds`.
- Output tổng hợp: `experiments/results_fs/fs_summary.csv` với
  mean/std/best của accuracy, fitness, n_selected_features cho mỗi
  cặp (algorithm, dataset).

**ĐIỂM DỪNG PHASE 3:** Báo cáo `fs_summary.csv`. Kiểm tra: ECL-SCSO có nằm
trong top 3 thuật toán tốt nhất ở đa số dataset không? Báo cáo cụ thể số
dataset ECL-SCSO thắng/thua so với SCSO gốc và baseline khác.

---

## 6. PHASE 4 — THỐNG KÊ & ABLATION STUDY

### 6.1. `statistical_tests.py`
- **Wilcoxon rank-sum test**: so ECL-SCSO với từng thuật toán còn lại, trên
  từng dataset/hàm benchmark riêng. Output bảng p-value, đánh dấu "+/-/="
  (ECL-SCSO thắng/thua/ngang theo ý nghĩa thống kê α=0.05).
- **Friedman test + average ranking**: xếp hạng tất cả 10 thuật toán qua
  toàn bộ dataset/hàm benchmark. Output bảng ranking trung bình.
- Dùng thư viện `scipy.stats` (wilcoxon) và `scikit-posthocs` hoặc tự code
  Friedman test.

### 6.2. Ablation Study
Chạy lại ECL-SCSO với các cấu hình:
1. Full (cả 4 cải tiến)
2. Chỉ Chaotic Init
3. Chỉ Adaptive R
4. Chỉ DE Mutation
5. Chỉ Levy Flight
6. Không cải tiến nào (= SCSO gốc, để đối chiếu)

Trên bộ benchmark function (không cần chạy lại hết 18 dataset, chỉ cần
5-6 hàm benchmark đại diện để tiết kiệm thời gian) — output
`experiments/results_benchmark/ablation_results.csv`.

---

## 7. PHASE 5 — TRỰC QUAN HÓA (`visualization/plots.py`)

Xuất các hình sau, lưu `.png` (300 dpi) VÀ `.pdf` (vector, để chèn LaTeX/Word
không bị vỡ nét) vào `experiments/figures/`:

1. **Convergence curves**: 1 hình/dataset hoặc hàm benchmark, các đường so
   sánh ECL-SCSO vs top 4-5 baseline mạnh nhất (không cần vẽ hết 10 đường
   gây rối hình).
2. **Boxplot**: phân phối fitness/accuracy của 30 lần chạy, theo từng
   thuật toán, cho mỗi dataset.
3. **Heatmap ranking**: ma trận algorithm × dataset, màu theo ranking
   (Friedman), giúp nhìn tổng quan ai thắng ở đâu.
4. **Bar chart**: số đặc trưng được chọn trung bình (selected feature ratio)
   theo từng thuật toán, từng dataset.
5. **Ablation bar chart**: so sánh fitness trung bình giữa 6 cấu hình
   ablation ở Phase 4.

**Style yêu cầu:** dùng `matplotlib` + `seaborn`, font dễ đọc, không dùng
màu mặc định lòe loẹt, legend rõ ràng, có thể bị reviewer Q1 soi kỹ —
ưu tiên màu sắc nhất quán xuyên suốt tất cả hình (cùng 1 thuật toán = cùng
1 màu ở mọi hình).

---

## 8. YÊU CẦU CHUNG XUYÊN SUỐT

1. **Không tự ý đổi tham số** trong `config.py` để "ra kết quả đẹp hơn".
   Nếu kết quả không như mong đợi, báo cáo trung thực và đề xuất hướng debug.
2. **Dừng lại ở mỗi "ĐIỂM DỪNG PHASE"** nêu trên, báo cáo kết quả, đợi tôi
   xác nhận trước khi chạy phase tiếp theo (vì mỗi phase đều tốn thời gian
   tính toán, tôi muốn kiểm tra trước khi để máy chạy qua đêm).
3. **Viết log tiến trình** ra console (ví dụ `tqdm` progress bar) cho mọi
   loop chạy 30 lần × nhiều thuật toán × nhiều dataset, vì thời gian chạy
   có thể vài giờ.
4. **Code phải chạy thuần CPU**, kiểm tra không có dependency nào âm thầm
   yêu cầu GPU/CUDA.
5. **Comment rõ công thức toán học** ở mọi nơi implement formula (để tôi
   đối chiếu khi viết Methodology của bài báo).
6. **Mọi random phải seed được** — không có yếu tố ngẫu nhiên không kiểm
   soát giữa các lần chạy lại.
7. Viết `README.md` hướng dẫn cách chạy lại toàn bộ pipeline từ đầu
   (`pip install -r requirements.txt`, sau đó các lệnh chạy từng phase).