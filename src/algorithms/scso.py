
from __future__ import annotations

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer


class SCSO(BaseOptimizer):

    def __init__(
        self,
        obj_func,
        dim: int,
        lb,
        ub,
        pop_size: int,
        max_iter: int,
        seed: int,
        S_M: float = 2.0,
    ) -> None:
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter, seed)
        self.S_M = S_M

    def _sensitivity_range(self, t: int) -> float:
        return self.S_M - self.S_M * t / self.max_iter

    def _scso_move_step(
        self,
        rG: float,
        population: np.ndarray,
        fitness: np.ndarray,
        best_solution: np.ndarray,
    ) -> None:
        for i in range(self.pop_size):
            R = 2.0 * rG * self.rng.random() - rG           
            r = rG * self.rng.random()           

            if abs(R) > 1.0:
                                       
                rand_idx = self.rng.integers(self.pop_size)
                P_rand = population[rand_idx]
                r3 = self.rng.random()
                new_pos = r * (P_rand - r3 * population[i])
            else:
                                            
                r4 = self.rng.random()
                Pr = r4 * best_solution - population[i]
                theta = self.rng.uniform(0.0, 2.0 * np.pi)
                new_pos = best_solution - r * Pr * np.cos(theta)

            population[i] = self._clip(new_pos)
            fitness[i] = self.obj_func(population[i])

    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:
        population = self._init_population()
        fitness = self._evaluate_population(population)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        convergence_curve = []

        for t in range(self.max_iter):
            rG = self._sensitivity_range(t)
            self._scso_move_step(rG, population, fitness, best_solution)

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_solution = population[gen_best_idx].copy()

            convergence_curve.append(best_fitness)

        return {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "convergence_curve": convergence_curve,
        }
