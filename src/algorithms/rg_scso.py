
from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.scso import SCSO
from src.feature_selection.relevance import RelevanceField, relevance_prior
from src.feature_selection.transfer_function import binarize_relevance, v_shaped


class RGSCSO(SCSO):

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        pop_size: int,
        max_iter: int,
        seed: int,
        X: np.ndarray,
        y: np.ndarray,
        eval_mask: Callable[[np.ndarray], float],
        S_M: float = 2.0,
        use_rms: bool = True,
        use_orl: bool = False,                                                       
        use_umr: bool = True,
        gamma: float = 0.5,
        umr_k: int = 8,
        umr_every: int = 1,
        ema_lambda: float = 0.9,
        w_online: float = 0.3,
        delta_scale: float = 0.01,
        max_nfe: int | None = None,
        prior_method: str = "mi",
        record_history: bool = False,
        tau: float = 0.5,
    ) -> None:
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter, seed, S_M)
        self.X = X
        self.y = y
        self.eval_mask = eval_mask
        self.use_rms = use_rms
        self.use_orl = use_orl
        self.use_umr = use_umr
        self.gamma = gamma
        self.tau = tau
        self.umr_k = umr_k
        self.umr_every = umr_every
        self.max_nfe = max_nfe if max_nfe is not None else pop_size * max_iter
        self._nfe = 0
                                                                                    
                                                                                  
        self.record_history = record_history
        self.history: list[dict] = []

        rho_static = relevance_prior(X, y, seed, method=prior_method)
        self.relevance = RelevanceField(
            rho_static, ema_lambda=ema_lambda, w_online=w_online, delta_scale=delta_scale
        )

                                                                                
    def _eval(self, mask: np.ndarray) -> float:
        self._nfe += 1
        return self.eval_mask(mask)

    def _rho(self) -> np.ndarray:
        return self.relevance.combined() if self.use_orl else self.relevance.rho_static

    def _binarize(self, x: np.ndarray, prev_bit: np.ndarray) -> np.ndarray:
        if self.use_rms:
            return binarize_relevance(
                x, prev_bit, self._rho(), self.rng, self.gamma, self.tau
            )
                                                                                
        do_flip = self.rng.random(x.shape) < v_shaped(x)
        return np.where(do_flip, 1 - prev_bit, prev_bit).astype(int)

    def _memetic_refine(
        self, best_bits: np.ndarray, best_fit: float
    ) -> tuple[np.ndarray, float]:
        rho = self._rho()
        uncertainty = np.abs(rho - self.tau)                                                              
        candidates = np.argsort(uncertainty)[: self.umr_k]
        bits = best_bits.copy()
        fit = best_fit
        for j in candidates:
            if self._nfe >= self.max_nfe:                           
                break
            trial = bits.copy()
            trial[j] = 1 - trial[j]
            trial_fit = self._eval(trial)
            if trial_fit < fit:
                bits, fit = trial, trial_fit
        return bits, fit

                                                                                 
    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:
                                                                                 
                                                           
        pos = self._init_population()
        bits = (pos > 0.0).astype(int)                                 
        fitness = np.array([self._eval(b) for b in bits], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_bits = bits[best_idx].copy()
        best_pos = pos[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        convergence_curve = []


        nfe_curve = []


        for t in range(self.max_iter):
            if self._nfe >= self.max_nfe:
                break
            rG = self._sensitivity_range(t)                           
            for i in range(self.pop_size):
                if self._nfe >= self.max_nfe:
                    break
                                                                                
                R = 2.0 * rG * self.rng.random() - rG
                r = rG * self.rng.random()
                if abs(R) > 1.0:
                    rand_idx = self.rng.integers(self.pop_size)
                    new_pos = r * (pos[rand_idx] - self.rng.random() * pos[i])
                else:
                    r4 = self.rng.random()
                    pr = r4 * best_pos - pos[i]
                    theta = self.rng.uniform(0.0, 2.0 * np.pi)
                    new_pos = best_pos - r * pr * np.cos(theta)
                new_pos = self._clip(new_pos)

                                                          
                new_bits = self._binarize(new_pos, bits[i])
                new_fit = self._eval(new_bits)

                                                               
                if self.use_orl:
                    self.relevance.update(new_bits, fitness[i] - new_fit)

                                                                        
                pos[i], bits[i], fitness[i] = new_pos, new_bits, new_fit

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_bits = bits[gen_best_idx].copy()
                best_pos = pos[gen_best_idx].copy()

                                                             
            if self.use_umr and t % self.umr_every == 0:
                best_bits, best_fitness = self._memetic_refine(best_bits, best_fitness)

            convergence_curve.append(best_fitness)
            nfe_curve.append(self._nfe)

            if self.record_history:


                p = bits.mean(axis=0)
                self.history.append({
                    "iter": t,
                    "diversity": float(np.mean(2.0 * p * (1.0 - p))),
                    "frozen_frac": float(np.mean((p == 0.0) | (p == 1.0))),
                    "mean_subset_size": float(bits.sum(axis=1).mean()),
                })


        if convergence_curve:
            convergence_curve.extend(
                [convergence_curve[-1]] * (self.max_iter - len(convergence_curve))
            )
        if nfe_curve:
            nfe_curve.extend([nfe_curve[-1]] * (self.max_iter - len(nfe_curve)))

        return {
            "best_solution": best_pos,
            "best_mask": best_bits,
            "best_fitness": best_fitness,
            "convergence_curve": convergence_curve,
            "nfe_curve": nfe_curve,
        }
