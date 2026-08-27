"""Hàm fitness cho binary feature selection (PROJECT_SPEC.md mục 5.3):

    fitness = ALPHA * error_rate + BETA * (n_selected_features / n_total_features)

    error_rate = 1 - accuracy, accuracy đo bằng KNN (k=KNN_NEIGHBORS) với
    k-fold cross-validation (k=KFOLD, scikit-learn).

    Nếu solution chọn 0 feature -> fitness = 1 (giá trị xấu nhất, tránh lỗi
    chia 0 / lựa chọn vô nghĩa).

LƯU Ý QUAN TRỌNG (tránh data leakage): StandardScaler được fit RIÊNG trên
từng fold training (qua sklearn Pipeline), KHÔNG fit trên toàn bộ dataset
trước khi chia fold — nếu không sẽ rò rỉ thông tin từ tập test vào quá trình
chuẩn hóa, một lỗi phổ biến trong nhiều bài báo cùng dạng.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FITNESS_ALPHA, FITNESS_BETA, KFOLD, KNN_NEIGHBORS
from src.feature_selection.transfer_function import (
    binarize_stochastic,
    binarize_threshold,
)

# Vài dataset (Zoo, Lymphography) có class hiếm với < KFOLD mẫu — sklearn
# vẫn xử lý đúng (giảm chất lượng stratify ở vài fold) nhưng cảnh báo NÀY lặp
# lại hàng triệu lần trong suốt Phase 3 (mỗi lần gọi obj_func), làm phình log
# vô ích. Tắt riêng cảnh báo này, KHÔNG tắt warning khác.
warnings.filterwarnings(
    "ignore", message="The least populated class in y has only .* members", category=UserWarning
)


def _cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    kfold: int,
    k_neighbors: int,
    seed: int,
    clf_factory: Callable[[], object] | None = None,
) -> float:
    """Stratified k-fold CV accuracy của 1 subset feature đã chọn.

    Scaler được fit RIÊNG mỗi fold (qua Pipeline) để tránh leakage. `clf_factory`
    (nếu có) trả về 1 estimator MỚI cho mỗi fold — cho phép hoán classifier
    (SVM/RF) mà giữ nguyên giao thức CV; None -> KNN(k=k_neighbors) mặc định, đúng
    hành vi gốc bit-for-bit.
    """
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    accuracies = []
    for train_idx, test_idx in skf.split(X, y):
        clf = clf_factory() if clf_factory is not None else KNeighborsClassifier(
            n_neighbors=k_neighbors
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X[train_idx], y[train_idx])
        accuracies.append(pipe.score(X[test_idx], y[test_idx]))
    return float(np.mean(accuracies))


def evaluate_binary_mask(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kfold: int = KFOLD,
    k_neighbors: int = KNN_NEIGHBORS,
    alpha: float = FITNESS_ALPHA,
    beta: float = FITNESS_BETA,
    seed: int = 42,
    clf_factory: Callable[[], object] | None = None,
) -> dict:
    """Tính fitness + accuracy + n_selected_features cho 1 binary mask đã chốt.

    `clf_factory` None -> KNN mặc định (hành vi gốc); truyền vào để đánh giá
    bằng classifier khác (SVM/RF) trong thí nghiệm robustness.
    """
    n_total = X.shape[1]
    selected_idx = np.flatnonzero(mask)
    n_selected = int(len(selected_idx))

    if n_selected == 0:
        return {
            "fitness": 1.0,
            "accuracy": 0.0,
            "n_selected_features": 0,
            "n_total_features": n_total,
        }

    accuracy = _cv_accuracy(X[:, selected_idx], y, kfold, k_neighbors, seed, clf_factory)
    error_rate = 1.0 - accuracy
    fitness = alpha * error_rate + beta * (n_selected / n_total)
    return {
        "fitness": fitness,
        "accuracy": accuracy,
        "n_selected_features": n_selected,
        "n_total_features": n_total,
    }


def make_fitness_function(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    kfold: int = KFOLD,
    k_neighbors: int = KNN_NEIGHBORS,
    alpha: float = FITNESS_ALPHA,
    beta: float = FITNESS_BETA,
    clf_factory: Callable[[], object] | None = None,
) -> Callable[[np.ndarray], float]:
    """Trả về obj_func(x_continuous) -> fitness, dùng trực tiếp làm `obj_func`
    cho BaseOptimizer. Binarize theo `binarize_stochastic` (Eq. mục 5.2) —
    RNG ring riêng, seed cố định để vẫn reproducible dù gọi obj_func nhiều lần
    với thứ tự khác nhau giữa các thuật toán.

    `clf_factory` None -> KNN mặc định (hành vi gốc); truyền vào để wrapper dùng
    classifier khác (SVM/RF) — phải NHẤT QUÁN với classifier dùng ở `eval_mask`.
    """
    rng = np.random.default_rng(seed)

    def obj_func(x: np.ndarray) -> float:
        mask = binarize_stochastic(x, rng)
        result = evaluate_binary_mask(
            mask, X, y, kfold, k_neighbors, alpha, beta, seed, clf_factory
        )
        return result["fitness"]

    return obj_func


def finalize_solution(
    x_continuous: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kfold: int = KFOLD,
    k_neighbors: int = KNN_NEIGHBORS,
    alpha: float = FITNESS_ALPHA,
    beta: float = FITNESS_BETA,
    seed: int = 42,
) -> dict:
    """Chốt feature subset CUỐI CÙNG từ best_solution liên tục của optimizer,
    dùng `binarize_threshold` (deterministic, DIM_BINARY_THRESHOLD) để kết
    quả báo cáo ổn định, không phụ thuộc lần random() cuối trong quá trình
    tìm kiếm.
    """
    mask = binarize_threshold(x_continuous)
    return evaluate_binary_mask(mask, X, y, kfold, k_neighbors, alpha, beta, seed)
