
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class BaseOptimizer(ABC):

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        pop_size: int,
        max_iter: int,
        seed: int,
    ) -> None:
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _init_population(self) -> np.ndarray:
        return self.lb + self.rng.random((self.pop_size, self.dim)) * (self.ub - self.lb)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.obj_func(ind) for ind in population], dtype=float)

    @abstractmethod
    def optimize(self) -> dict:
        raise NotImplementedError

    def _timed(self, run_fn: Callable[[], dict]) -> dict:
        start = time.perf_counter()
        result = run_fn()
        result["runtime"] = time.perf_counter() - start
        return result
