"""ECL-SCSO — Enhanced Chaotic-Levy Sand Cat Swarm Optimization (thuật toán đề xuất).

Kế thừa `SCSO` (giữ nguyên công thức di chuyển gốc Eq. (4)-(8), xem
`scso.py`) và bổ sung 4 cải tiến độc lập, mỗi cải tiến bật/tắt được qua flag
constructor để phục vụ ablation study (Phase 4):

Cải tiến 1 — Chaotic Initialization (Tent map):
    x(n+1) = x(n)/0.7                  nếu x(n) < 0.7
    x(n+1) = (10/3) * x(n) * (1-x(n))  nếu x(n) >= 0.7
    Sinh quần thể khởi tạo bằng dãy tent-map (rồi scale về [lb, ub]) thay vì
    uniform random thuần túy, để cải thiện độ phủ không gian tìm kiếm.

Cải tiến 2 — Adaptive Sensitivity Range (phi tuyến, thay cho Eq. (3) tuyến tính):
    R(t) = R_max * cos(pi/2 * t / max_iter)

Cải tiến 3 — DE Mutation cho cá thể yếu:
    Mỗi vòng lặp, lấy `de_mutation_ratio` (mặc định 30%) cá thể có fitness
    tệ nhất. Với mỗi cá thể i trong nhóm này:
        V_i = X_r1 + de_F * (X_r2 - X_r3)      (mutation, r1≠r2≠r3≠i)
        U_i = binomial_crossover(V_i, X_i, CR=0.7)
    Nếu fitness(U_i) tốt hơn X_i -> thay thế.

Cải tiến 4 — Levy Flight khi trì trệ:
    stagnation_counter tăng 1 mỗi vòng lặp best fitness KHÔNG cải thiện,
    reset về 0 khi có cải thiện. Khi stagnation_counter >= stagnation_threshold:
        step = Levy(lambda=1.5)   # Mantegna's algorithm
        X_new = X_best + step * (X_best - X_random_other)
    Nếu X_new tốt hơn best hiện tại -> cập nhật, reset stagnation_counter.

LƯU Ý: convergence_curve lưu best fitness ở MỖI vòng lặp (độ dài = max_iter).
"""

from __future__ import annotations

import math

import numpy as np

from src.algorithms.scso import SCSO


class ECLSCSO(SCSO):
    """SCSO + 4 cải tiến (chaotic init, adaptive R, DE mutation, Levy flight)."""

    def __init__(
        self,
        obj_func,
        dim: int,
        lb,
        ub,
        pop_size: int,
        max_iter: int,
        seed: int,
        use_chaotic_init: bool = True,
        use_adaptive_R: bool = True,
        use_de_mutation: bool = True,
        use_levy_flight: bool = True,
        stagnation_threshold: int = 10,
        de_mutation_ratio: float = 0.3,
        de_F: float = 0.5,
        de_CR: float = 0.7,
        S_M: float = 2.0,
        levy_beta: float = 1.5,
    ) -> None:
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter, seed, S_M=S_M)
        self.use_chaotic_init = use_chaotic_init
        self.use_adaptive_R = use_adaptive_R
        self.use_de_mutation = use_de_mutation
        self.use_levy_flight = use_levy_flight
        self.stagnation_threshold = stagnation_threshold
        self.de_mutation_ratio = de_mutation_ratio
        self.de_F = de_F
        self.de_CR = de_CR
        self.levy_beta = levy_beta

    # ------------------------------------------------------------------
    # Cải tiến 1 — Chaotic Initialization (Tent map)
    # ------------------------------------------------------------------
    def _tent_map_sequence(self) -> np.ndarray:
        sequence = np.empty((self.pop_size, self.dim))
        x = self.rng.random(self.dim)
        x = np.clip(x, 1e-9, 1.0 - 1e-9)  # tránh điểm bất động x=0
        for i in range(self.pop_size):
            x = np.where(x < 0.7, x / 0.7, (10.0 / 3.0) * x * (1.0 - x))
            x = np.clip(x, 1e-9, 1.0 - 1e-9)  # tránh trôi ra ngoài [0,1] do sai số số học
            sequence[i] = x
        return sequence

    def _init_population(self) -> np.ndarray:
        if not self.use_chaotic_init:
            return super()._init_population()
        tent_sequence = self._tent_map_sequence()
        return self.lb + tent_sequence * (self.ub - self.lb)

    # ------------------------------------------------------------------
    # Cải tiến 2 — Adaptive Sensitivity Range (phi tuyến)
    # ------------------------------------------------------------------
    def _sensitivity_range(self, t: int) -> float:
        if not self.use_adaptive_R:
            return super()._sensitivity_range(t)
        return self.S_M * math.cos((math.pi / 2.0) * t / self.max_iter)

    # ------------------------------------------------------------------
    # Cải tiến 3 — DE Mutation cho cá thể yếu
    # ------------------------------------------------------------------
    def _de_mutate_worst(self, population: np.ndarray, fitness: np.ndarray) -> None:
        n_worst = max(1, int(round(self.de_mutation_ratio * self.pop_size)))
        worst_indices = np.argsort(fitness)[-n_worst:]  # fitness cao nhất = tệ nhất (minimization)
        all_indices = np.arange(self.pop_size)

        for i in worst_indices:
            candidates = all_indices[all_indices != i]
            r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)
            V = population[r1] + self.de_F * (population[r2] - population[r3])

            j_rand = self.rng.integers(self.dim)
            cross_mask = self.rng.random(self.dim) < self.de_CR
            cross_mask[j_rand] = True  # đảm bảo ít nhất 1 chiều được lấy từ V (binomial crossover)
            trial = np.where(cross_mask, V, population[i])
            trial = self._clip(trial)

            trial_fitness = self.obj_func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

    # ------------------------------------------------------------------
    # Cải tiến 4 — Levy Flight khi trì trệ (Mantegna's algorithm)
    # ------------------------------------------------------------------
    def _levy_step(self) -> np.ndarray:
        beta = self.levy_beta
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1 + beta) / 2.0) * beta * 2.0 ** ((beta - 1) / 2.0))
        ) ** (1.0 / beta)
        u = self.rng.normal(0.0, sigma_u, size=self.dim)
        v = self.rng.normal(0.0, 1.0, size=self.dim)
        return u / np.abs(v) ** (1.0 / beta)

    def _levy_flight_escape(
        self, population: np.ndarray, best_solution: np.ndarray, best_fitness: float
    ) -> tuple[np.ndarray, float] | None:
        step = self._levy_step()
        rand_idx = self.rng.integers(self.pop_size)
        X_random_other = population[rand_idx]
        X_new = self._clip(best_solution + step * (best_solution - X_random_other))
        fit_new = float(self.obj_func(X_new))
        if fit_new < best_fitness:
            return X_new, fit_new
        return None

    # ------------------------------------------------------------------
    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:
        population = self._init_population()
        fitness = self._evaluate_population(population)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        convergence_curve = []
        stagnation_counter = 0

        for t in range(self.max_iter):
            rG = self._sensitivity_range(t)
            self._scso_move_step(rG, population, fitness, best_solution)

            if self.use_de_mutation:
                self._de_mutate_worst(population, fitness)

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_solution = population[gen_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if self.use_levy_flight and stagnation_counter >= self.stagnation_threshold:
                escaped = self._levy_flight_escape(population, best_solution, best_fitness)
                if escaped is not None:
                    best_solution, best_fitness = escaped
                    stagnation_counter = 0

            convergence_curve.append(best_fitness)

        return {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "convergence_curve": convergence_curve,
        }
