# Baseline Fidelity & Tuning Report — RG-SCSO (IJCS)

> **Mục đích (SOP 1 Q5 / F1):** document hoá đầy đủ config/tuning/seed/budget của **mọi** thuật toán so sánh, để chứng minh so sánh là **faithful published-default comparison dưới một protocol đồng nhất**, không phải "tuned method vs library-default baseline".
> **Nguồn:** trích trực tiếp từ code (`config.py`, `src/feature_selection/run_fs_heldout.py`, `src/algorithms/baselines.py`, `src/algorithms/rg_scso.py`, `src/algorithms/scso.py`), commit `246b142`. Không có số liệu gõ tay.

## 1. Protocol dùng chung cho tất cả 7 thuật toán

| Thành phần | Giá trị | Nguồn |
|---|---|---|
| Population size | 30 | `config.POPULATION_SIZE` |
| Iterations / epochs | 500 | `config.MAX_ITERATION` |
| Independent runs | 30 | `config.NUM_INDEPENDENT_RUNS` |
| Seed protocol | `seed = 42 + run_id`, **paired across algorithms** | `config.RANDOM_SEED_BASE`; mealpy `model.solve(problem, seed=seed)`; RGSCSO/SCSO `seed=seed` |
| Search space | continuous `[-1, 1]^d` | `run_fs_heldout.SEARCH_LB/UB = -1.0/1.0` |
| Objective (fitness) | `0.99·(1−Acc) + 0.01·|b|/d`, Acc = stratified 5-fold KNN (k=5) trên **training partition** | `make_fitness_function`, `config.KFOLD=5`, `KNN_NEIGHBORS=5` |
| Binarization (baselines) | threshold 0.5 (`binarize_threshold`) | `config.DIM_BINARY_THRESHOLD=0.5` |
| NFE budget | pop×iter; RG-SCSO **hard-capped = 15,000**; mealpy epoch-based = 30×501 = **15,030** | `rg_scso.py` hard cap `_nfe>=max_nfe`; mealpy convention |
| Held-out | outer 80/20 stratified split; prior+search+CV chỉ trên train | `run_fs_heldout.py` |

**Không thuật toán nào được tune hyperparameter per-dataset.** Không có DEV/VAL hyperparameter search trong code cho bất kỳ method nào.

## 2. Bảng fidelity per-method (SOP 1 Q5 required table)

| Method | Source / class | Internal hyperparameters | Search space | #Trials | Tuning budget | Seed | Reproducibility |
|---|---|---|---|---|---|---|---|
| **RG-SCSO** (proposed) | `src/algorithms/rg_scso.py` | γ=0.5, K(umr)=8, τ=0.5, S_M=2.0 — **hằng số cố định**; MI prior (Kraskov) | [-1,1]^d | 30 runs | **none per-dataset** (γ/K/τ fixed pre-run; có sensitivity sweep τ,λ ở Supp) | 42+run_id | code + seeds + raw CSV |
| SCSO (base) | `src/algorithms/scso.py` | S_M=2.0 (published) | [-1,1]^d | 30 | none | 42+run_id | ✓ |
| AOA | mealpy `OriginalAOA` | **library default** (α, μ, …) | [-1,1]^d | 30 | none | 42+run_id | ✓ |
| CoatiOA (COA) | mealpy `OriginalCoatiOA` | library default | [-1,1]^d | 30 | none | 42+run_id | ✓ |
| GWO | mealpy `OriginalGWO` | library default (a: 2→0) | [-1,1]^d | 30 | none | 42+run_id | ✓ |
| PSO | mealpy `OriginalPSO` | library default (c1,c2,w) | [-1,1]^d | 30 | none | 42+run_id | ✓ |
| RIME | mealpy `OriginalRIME` | library default | [-1,1]^d | 30 | none | 42+run_id | ✓ |

`model_cls(epoch=500, pop_size=30)` cho mọi mealpy baseline — chỉ đặt epoch/pop, còn lại default (`baselines.py:64`).

## 3. Kết luận fairness (option B — faithful reproduction)

Theo khung SOP 1 Q5, đây là **Option B — faithful published-default comparison**, được chọn minh bạch (không phải Option A "fair tuning"):
- **Fair:** protocol tìm kiếm (population, iterations, budget, seeds, objective, evaluation, split) **đồng nhất tuyệt đối** giữa 7 thuật toán; seeds paired ⇒ mọi so sánh là paired.
- **Faithful:** mỗi thuật toán chạy ở **published/default hyperparameters** của chính nó; RG-SCSO's γ/K/τ là hằng số cố định trước run (được củng cố bởi sensitivity sweep τ,λ trong Supplementary), **không** tune per-dataset.
- **Không advantage ẩn về budget:** RG-SCSO bị cap ở đúng 15,000 NFE, **ít hơn 30 eval** so với baseline (15,030) — bất lợi, không phải lợi thế.

**Wording khuyến nghị cho manuscript:** gọi là *"a faithful comparison under an identical search protocol with each method at its published default hyperparameters"* — **không** dùng "fairly tuned comparison".

**Giới hạn tự thừa nhận:** vì baselines không được tune per-dataset, đây không phải "best-tuned-vs-best-tuned"; kết quả nên đọc là *RG-SCSO's mechanism dưới protocol chuẩn*, không phải "RG-SCSO thắng mọi baseline đã tune tối ưu".
