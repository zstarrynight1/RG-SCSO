
from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer

                                                                            
TVT_TAU_MAX = 4.0
TVT_TAU_MIN = 0.01


def islam_tvt_pflip(x: np.ndarray, t: int, max_iter: int) -> np.ndarray:
    frac = t / max_iter if max_iter > 0 else 0.0
    tau = TVT_TAU_MAX - (TVT_TAU_MAX - TVT_TAU_MIN) * frac
    return np.abs(np.tanh(tau * x))


def teng_v4_pflip(x: np.ndarray, t: int, max_iter: int) -> np.ndarray:
    return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * x))


TransferFn = Callable[[np.ndarray, int, int], np.ndarray]


class _BinaryOptimizer(BaseOptimizer):

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        pop_size: int,
        max_iter: int,
        seed: int,
        eval_mask: Callable[[np.ndarray], float],
        transfer: TransferFn,
        max_nfe: int | None = None,
    ) -> None:
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter, seed)
        self.eval_mask = eval_mask
        self.transfer = transfer
        self.max_nfe = max_nfe if max_nfe is not None else pop_size * max_iter
        self._nfe = 0

    def _eval(self, mask: np.ndarray) -> float:
        self._nfe += 1
        return self.eval_mask(mask)

    def _binarize(self, x: np.ndarray, t: int, prev_bit: np.ndarray) -> np.ndarray:
        p_flip = np.clip(self.transfer(x, t, self.max_iter), 0.0, 1.0)
        do_flip = self.rng.random(x.shape) < p_flip
        return np.where(do_flip, 1 - prev_bit, prev_bit).astype(int)

    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:  # pragma: no cover - abstract-ish
        raise NotImplementedError


class BinaryPSO(_BinaryOptimizer):

    def __init__(self, *args, w_max: float = 0.9, w_min: float = 0.4,
                 c1: float = 2.0, c2: float = 2.0, v_max: float = 6.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.v_max = v_max

    def _run(self) -> dict:
                                                                  
        pos = self._init_population()
        bits = (pos > 0.0).astype(int)
        vel = np.zeros((self.pop_size, self.dim), dtype=float)
        fitness = np.array([self._eval(b) for b in bits], dtype=float)

        pbest_bits = bits.copy()
        pbest_fit = fitness.copy()
        g_idx = int(np.argmin(fitness))
        gbest_bits = bits[g_idx].copy()
        gbest_fit = float(fitness[g_idx])
        convergence_curve = []

        for t in range(self.max_iter):
            if self._nfe >= self.max_nfe:
                break
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter
            for i in range(self.pop_size):
                if self._nfe >= self.max_nfe:
                    break
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                vel[i] = (
                    w * vel[i]
                    + self.c1 * r1 * (pbest_bits[i] - bits[i])
                    + self.c2 * r2 * (gbest_bits - bits[i])
                )
                vel[i] = np.clip(vel[i], -self.v_max, self.v_max)
                bits[i] = self._binarize(vel[i], t, bits[i])
                fitness[i] = self._eval(bits[i])

                if fitness[i] < pbest_fit[i]:
                    pbest_fit[i] = fitness[i]
                    pbest_bits[i] = bits[i].copy()
                if fitness[i] < gbest_fit:
                    gbest_fit = float(fitness[i])
                    gbest_bits = bits[i].copy()

            convergence_curve.append(gbest_fit)

        if convergence_curve:
            convergence_curve.extend(
                [convergence_curve[-1]] * (self.max_iter - len(convergence_curve))
            )
        return {
            "best_solution": np.where(gbest_bits > 0, 1.0, -1.0),
            "best_mask": gbest_bits,
            "best_fitness": gbest_fit,
            "convergence_curve": convergence_curve,
        }


class BinaryGWO(_BinaryOptimizer):

    def _gwo_move(self, a: float, pos_i: np.ndarray,
                  alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray) -> np.ndarray:
        new = np.empty(self.dim, dtype=float)
        for lead in (alpha, beta, delta):
            r1 = self.rng.random(self.dim)
            r2 = self.rng.random(self.dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * lead - pos_i)
            new += lead - A * D
        return new / 3.0

    def _run(self) -> dict:
        pos = self._init_population()
        bits = (pos > 0.0).astype(int)
        fitness = np.array([self._eval(b) for b in bits], dtype=float)

        order = np.argsort(fitness)
        alpha_pos, beta_pos, delta_pos = (pos[order[k]].copy() for k in range(3))
        alpha_bits = bits[order[0]].copy()
        alpha_fit = float(fitness[order[0]])
        beta_fit, delta_fit = float(fitness[order[1]]), float(fitness[order[2]])
        convergence_curve = []

        for t in range(self.max_iter):
            if self._nfe >= self.max_nfe:
                break
            a = 2.0 - 2.0 * t / self.max_iter
            for i in range(self.pop_size):
                if self._nfe >= self.max_nfe:
                    break
                new_pos = self._clip(self._gwo_move(a, pos[i], alpha_pos, beta_pos, delta_pos))
                new_bits = self._binarize(new_pos, t, bits[i])
                new_fit = self._eval(new_bits)
                pos[i], bits[i], fitness[i] = new_pos, new_bits, new_fit

                                                                              
                if new_fit < alpha_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = alpha_pos, alpha_fit
                    alpha_pos, alpha_bits, alpha_fit = new_pos.copy(), new_bits.copy(), new_fit
                elif new_fit < beta_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = new_pos.copy(), new_fit
                elif new_fit < delta_fit:
                    delta_pos, delta_fit = new_pos.copy(), new_fit

            convergence_curve.append(alpha_fit)

        if convergence_curve:
            convergence_curve.extend(
                [convergence_curve[-1]] * (self.max_iter - len(convergence_curve))
            )
        return {
            "best_solution": np.where(alpha_bits > 0, 1.0, -1.0),
            "best_mask": alpha_bits,
            "best_fitness": alpha_fit,
            "convergence_curve": convergence_curve,
        }
