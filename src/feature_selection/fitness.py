
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
    mask = binarize_threshold(x_continuous)
    return evaluate_binary_mask(mask, X, y, kfold, k_neighbors, alpha, beta, seed)
