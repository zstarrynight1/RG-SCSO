
from __future__ import annotations

import numpy as np

from config import DIM_BINARY_THRESHOLD


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binarize_stochastic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = sigmoid(x)
    return (rng.random(x.shape) < s).astype(int)


def binarize_threshold(x: np.ndarray, threshold: float = DIM_BINARY_THRESHOLD) -> np.ndarray:
    s = sigmoid(x)
    return (s >= threshold).astype(int)


def s_shaped(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)


def v_shaped(x: np.ndarray) -> np.ndarray:
    return np.abs(np.tanh(x))


def binarize_relevance(
    x: np.ndarray,
    prev_bit: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
    gamma: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    p_flip = v_shaped(x)
    preferred = (rho > threshold).astype(int)
    norm = max(threshold, 1.0 - threshold)
    strength = np.abs(rho - threshold) / norm
    flip_bit = 1 - prev_bit
    toward_preferred = flip_bit == preferred
    factor = np.where(toward_preferred, 1.0 + gamma * strength, 1.0 - gamma * strength)
    p_flip = np.clip(p_flip * factor, 0.0, 1.0)
    do_flip = rng.random(x.shape) < p_flip
    return np.where(do_flip, flip_bit, prev_bit).astype(int)
