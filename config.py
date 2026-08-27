"""Tham số tập trung cho toàn bộ pipeline ECL-SCSO.

Mọi script chạy thí nghiệm import từ đây, KHÔNG hardcode rải rác trong code.
"""

# --- Tham số chung cho optimizer ---
POPULATION_SIZE = 30
MAX_ITERATION = 500
NUM_INDEPENDENT_RUNS = 30  # mỗi thuật toán chạy 30 lần độc lập
RANDOM_SEED_BASE = 42  # seed = RANDOM_SEED_BASE + run_index, để reproducible

# --- Tham số feature selection / fitness ---
KFOLD = 5  # k-fold cross-validation cho fitness
KNN_NEIGHBORS = 5
FITNESS_ALPHA = 0.99  # trọng số error rate
FITNESS_BETA = 0.01  # trọng số tỉ lệ feature được chọn
DIM_BINARY_THRESHOLD = 0.5  # ngưỡng sigmoid để binarize

# --- Tham số riêng của ECL-SCSO (giá trị mặc định, có thể override khi gọi) ---
SCSO_S_M = 2.0  # sensitivity range tối đa (rG ban đầu / R_max)
ECL_STAGNATION_THRESHOLD = 10
ECL_DE_MUTATION_RATIO = 0.3
ECL_DE_F = 0.5
ECL_DE_CR = 0.7
ECL_LEVY_BETA = 1.5
