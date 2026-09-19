
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


PRIOR_METHODS = ("mi", "relieff", "fisher", "shuffled_mi", "inverted_mi")


def relevance_prior(
    X: np.ndarray, y: np.ndarray, seed: int, method: str = "mi"
) -> np.ndarray:
    if method not in PRIOR_METHODS:
        raise ValueError(f"method phải thuộc {PRIOR_METHODS}, nhận '{method}'.")

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_y = float(-np.sum(p * np.log(p + 1e-12)))                       
    if h_y <= 0.0:                                                           
        return np.full(X.shape[1], 0.5)

    if method == "mi":
        mi = mutual_info_classif(X, y, random_state=seed)
        return np.clip(mi / h_y, 0.0, 1.0)
    if method == "relieff":
        return _relieff_prior(X, y, seed)
    if method == "fisher":
        return _fisher_prior(X, y)
    if method == "shuffled_mi":
        mi = mutual_info_classif(X, y, random_state=seed)
        rho = np.clip(mi / h_y, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        return rng.permutation(rho)


    mi = mutual_info_classif(X, y, random_state=seed)
    rho = np.clip(mi / h_y, 0.0, 1.0)
    return 1.0 - rho


def _minmax01(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.full_like(v, 0.5, dtype=float)
    return (v - lo) / (hi - lo)


def _fisher_prior(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    mu = X.mean(axis=0)
    num = np.zeros(X.shape[1], dtype=float)
    den = np.zeros(X.shape[1], dtype=float)
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        num += nc * (Xc.mean(axis=0) - mu) ** 2
        den += nc * Xc.var(axis=0)
    fisher = num / (den + 1e-12)
    return _minmax01(fisher)


def _relieff_prior(
    X: np.ndarray, y: np.ndarray, seed: int, n_neighbors: int = 10, max_samples: int = 300
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    span = X.max(axis=0) - X.min(axis=0)
    span[span < 1e-12] = 1.0
    Xn = (X - X.min(axis=0)) / span                                              

    classes, counts = np.unique(y, return_counts=True)
    prior = {c: cnt / n for c, cnt in zip(classes, counts)}

    m = min(max_samples, n)
    sample_idx = rng.choice(n, size=m, replace=False)
    k = max(1, min(n_neighbors, n - 1))

    w = np.zeros(d, dtype=float)
    for i in sample_idx:
        xi, yi = Xn[i], y[i]
        dist = np.abs(Xn - xi).sum(axis=1)                              
        dist[i] = np.inf
                                                                              
        same = np.where(y == yi)[0]
        same = same[same != i]
        if same.size:
            hits = same[np.argsort(dist[same])[:k]]
            w -= np.abs(Xn[hits] - xi).sum(axis=0) / (m * hits.size)
                                                                 
        denom = 1.0 - prior[yi]
        for c in classes:
            if c == yi:
                continue
            oth = np.where(y == c)[0]
            if oth.size == 0:
                continue
            miss = oth[np.argsort(dist[oth])[:k]]
            weight_c = (prior[c] / denom) if denom > 1e-12 else 0.0
            w += weight_c * np.abs(Xn[miss] - xi).sum(axis=0) / (m * miss.size)

    scale = float(np.max(np.abs(w)))
    if scale < 1e-12:
        return np.full(d, 0.5)
    return np.clip(0.5 + w / (2.0 * scale), 0.0, 1.0)


class RelevanceField:

    def __init__(
        self,
        rho_static: np.ndarray,
        ema_lambda: float = 0.9,
        w_online: float = 0.3,
        delta_scale: float = 0.01,
    ) -> None:
        self.rho_static = np.asarray(rho_static, dtype=float)
        self.online = np.zeros_like(self.rho_static)
        self.ema_lambda = ema_lambda
        self.w_online = w_online
        self.delta_scale = delta_scale

    def update(self, mask: np.ndarray, fitness_delta: float) -> None:
        reward = float(np.tanh(fitness_delta / self.delta_scale))
        signal = mask.astype(float) * reward                                        
        self.online = self.ema_lambda * self.online + (1.0 - self.ema_lambda) * signal

    def combined(self) -> np.ndarray:
        return np.clip(self.rho_static + self.w_online * self.online, 0.0, 1.0)
