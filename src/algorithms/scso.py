"""SCSO gốc — Sand Cat Swarm Optimization (Seyyedabbasi & Kiani, 2022).

LƯU Ý KHI ĐỐI CHIẾU METHODOLOGY:
Bài báo gốc ("Sand Cat swarm optimization: a nature-inspired algorithm to
solve global optimization problems", Engineering with Computers, 2022) nằm
sau paywall (Springer) nên không fetch trực tiếp được. Công thức dưới đây
được tổng hợp/đối chiếu chéo từ 2 nguồn học thuật độc lập trích dẫn lại
phương trình gốc (bài "Multi-Strategy Improved Sand Cat Swarm Optimization",
PMC10604673, và mô tả tổng quan của chính nhóm tác giả). Trước khi đưa vào
phần Methodology chính thức của bài báo, NÊN đối chiếu lại 1 lần nữa với PDF
gốc hoặc code MATLAB chính chủ trên Mathworks File Exchange
(fileexchange/110185-sand-cat-swarm-optimization).

Công thức cốt lõi:
    rG(t) = S_M - S_M * t / max_iter            # (3) sensitivity range tổng quát,
                                                  # giảm tuyến tính từ S_M (=2) về 0
    R     = 2 * rG * r1 - rG                     # (4) r1 ~ U(0,1), R quyết định pha
    r     = rG * r2                              # (5) r2 ~ U(0,1)

    Nếu |R| > 1  -> EXPLORATION (tìm kiếm mồi):
        P(t+1) = r * (P_rand(t) - r3 * P(t))     # (6) r3 ~ U(0,1),
                                                  # P_rand: 1 cá thể ngẫu nhiên trong quần thể

    Nếu |R| <= 1 -> EXPLOITATION (tấn công mồi, chuyển động xoắn ốc):
        Pr(t)    = r4 * P_best(t) - P(t)         # (7) r4 ~ U(0,1)
        P(t+1)   = P_best(t) - r * Pr(t) * cos(theta)   # (8)
                                                  # theta: góc ngẫu nhiên trong [0, 2*pi]
                                                  # (roulette-wheel theo bài gốc; ở đây
                                                  # dùng uniform random — cách phổ biến
                                                  # trong các bản re-implementation mở)

R, r, r3, r4, theta được lấy là số vô hướng cho MỖI cá thể MỖI vòng lặp (không
phải vector theo từng chiều) — đúng như ký hiệu vô hướng trong các phương
trình (3)-(8) ở trên.
"""

from __future__ import annotations

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer


class SCSO(BaseOptimizer):
    """SCSO gốc, chưa cải tiến — dùng làm baseline đối chiếu với ECL-SCSO."""

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
        """rG(t) — Eq. (3): giảm tuyến tính từ S_M về 0."""
        return self.S_M - self.S_M * t / self.max_iter

    def _scso_move_step(
        self,
        rG: float,
        population: np.ndarray,
        fitness: np.ndarray,
        best_solution: np.ndarray,
    ) -> None:
        """1 vòng di chuyển toàn quần thể theo Eq. (4)-(8). Sửa population/fitness in-place.

        Tách riêng method này (thay vì nhúng thẳng trong `_run`) để ECL-SCSO có
        thể tái sử dụng đúng công thức di chuyển gốc, chỉ thay đổi cách tính
        `rG` (Cải tiến 2 — adaptive sensitivity) và thêm các bước hậu xử lý
        (Cải tiến 3, 4).
        """
        for i in range(self.pop_size):
            R = 2.0 * rG * self.rng.random() - rG  # Eq. (4)
            r = rG * self.rng.random()  # Eq. (5)

            if abs(R) > 1.0:
                # Exploration — Eq. (6)
                rand_idx = self.rng.integers(self.pop_size)
                P_rand = population[rand_idx]
                r3 = self.rng.random()
                new_pos = r * (P_rand - r3 * population[i])
            else:
                # Exploitation — Eq. (7)-(8)
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
