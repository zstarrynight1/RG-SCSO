"""Binary baseline optimizers có TRANSFER FUNCTION THÍCH NGHI — để CÔ LẬP đóng
góp lõi của RG-SCSO (độ-liên-quan theo TỪNG feature), trả lời reviewer #1/Q4.

ĐỘNG CƠ (vì sao cần file này):
    RG-SCSO thắng nhờ HAI thứ chồng lên nhau: (i) binarize V-shaped (bảo toàn
    động lực nhị phân — chữa washout) VÀ (ii) điều biến ngưỡng lật theo trường
    độ-liên-quan ρ theo TỪNG feature (bơm tri thức bài toán). Một reviewer Q1 sẽ
    hỏi: liệu lợi thế có chỉ đến từ (i) — tức bất kỳ binarize V-shaped/thích nghi
    nào cũng đủ — hay thật sự cần (ii)? Để tách bạch, ta chạy HAI optimizer nhị
    phân kinh điển (PSO, GWO) trang bị các transfer function THÍCH NGHI / BIẾN
    THIÊN THỜI GIAN đã công bố, NHƯNG KHÔNG có relevance per-feature. Nếu RG-SCSO
    vẫn vượt chúng, lợi thế đến từ (ii) — chính là cái mới.

HAI HỌ TRANSFER (đều V-shaped, đều KHÔNG dùng relevance):
    - Islam TVT (time-varying): V_τ(x) = |tanh(τ·x)|, τ giảm tuyến tính 4 → 0.01
      qua các vòng lặp (explore mạnh sớm, exploit mạnh muộn). Nguồn: Islam, Li &
      Mei, Appl. Soft Comput. 59:182-196 (2017); hằng số τ_max=4, τ_min=0.01 theo
      bản tái lập chuẩn của Mafarja et al., Knowl.-Based Syst. 161:185-204 (2018).
    - Teng V4 (adaptive V-shaped BPSO): T(x) = |(2/π)·arctan((π/2)·x)|, cố định.
      Nguồn: Teng, Dong & Zhou, PLoS ONE 12(3):e0173907 (2017), transfer V4.

LƯU Ý SOURCING (giống caveat của SCSO trong scso.py):
    Cả hai họ dùng CHUNG luật lật V-shaped chuẩn Mirjalili & Lewis (2013): lật bit
    TƯƠNG ĐỐI so với bit hiện tại nếu rand() < p_flip. Teng (2017) trình bày một
    luật gán TUYỆT ĐỐI (x=0 nếu rand<T, x=1 nếu rand≥T) và phần "adaptive" của
    họ nằm ở KÍCH THƯỚC subset qua hàm fitness entropy — KHÔNG phải ở độ dốc
    transfer. Ở đây ta chỉ mượn HÌNH DẠNG transfer V4 của Teng và áp luật lật
    chuẩn dùng chung cho cả hai config, để khác biệt DUY NHẤT giữa các baseline là
    (a) optimizer nền (PSO/GWO) và (b) hình dạng transfer — cô lập sạch yếu tố
    relevance. Giao thức đánh giá (eval_mask, ngân sách NFE, biên, seed) KHỚP
    tuyệt đối với RG-SCSO để so sánh công bằng.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer

# Hằng số lịch trình τ của Islam TVT (theo bản tái lập Mafarja et al. 2018).
TVT_TAU_MAX = 4.0
TVT_TAU_MIN = 0.01


# --------------------------------------------------------------------------- #
# Họ transfer function V-shaped (KHÔNG relevance) — trả về XÁC SUẤT LẬT ∈ [0,1)
# --------------------------------------------------------------------------- #
def islam_tvt_pflip(x: np.ndarray, t: int, max_iter: int) -> np.ndarray:
    """Islam time-varying V-shaped: V_τ(x) = |tanh(τ(t)·x)|.

    τ(t) giảm tuyến tính từ τ_max=4 (explore) về τ_min=0.01 (exploit) theo vòng
    lặp t. τ lớn -> |tanh| bão hòa nhanh -> xác suất lật cao; τ nhỏ -> gần như
    đóng băng bit (khai thác).

    Args:
        x: Vị trí/vận tốc liên tục, shape (n_features,).
        t: Chỉ số vòng lặp hiện tại (0-based).
        max_iter: Tổng số vòng lặp (để tính lịch trình τ).

    Returns:
        Xác suất lật bit, np.ndarray shape (n_features,), giá trị ∈ [0, 1).
    """
    frac = t / max_iter if max_iter > 0 else 0.0
    tau = TVT_TAU_MAX - (TVT_TAU_MAX - TVT_TAU_MIN) * frac
    return np.abs(np.tanh(tau * x))


def teng_v4_pflip(x: np.ndarray, t: int, max_iter: int) -> np.ndarray:
    """Teng V4 (adaptive V-shaped BPSO): T(x) = |(2/π)·arctan((π/2)·x)|, cố định.

    Không phụ thuộc t (đối số t/max_iter giữ cho khớp interface với Islam TVT).

    Args:
        x: Vị trí/vận tốc liên tục, shape (n_features,).
        t, max_iter: Không dùng (giữ interface thống nhất).

    Returns:
        Xác suất lật bit, np.ndarray shape (n_features,), giá trị ∈ [0, 1).
    """
    return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * x))


# type alias: (x, t, max_iter) -> p_flip
TransferFn = Callable[[np.ndarray, int, int], np.ndarray]


class _BinaryOptimizer(BaseOptimizer):
    """Nền chung cho các optimizer nhị phân dùng eval_mask + ngân sách NFE.

    Giữ ĐÚNG khế ước công bằng của RG-SCSO: chấm điểm TRỰC TIẾP trên binary mask
    qua `eval_mask` (không threshold `best_solution` như baseline mealpy), và giới
    hạn tổng số lần đánh giá = pop_size × max_iter để không config nào được nhiều
    eval hơn.
    """

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
        """Chấm 1 mask, tăng bộ đếm NFE (mọi eval đều tính vào ngân sách)."""
        self._nfe += 1
        return self.eval_mask(mask)

    def _binarize(self, x: np.ndarray, t: int, prev_bit: np.ndarray) -> np.ndarray:
        """Luật lật V-shaped chuẩn: lật bit TƯƠNG ĐỐI nếu rand() < p_flip."""
        p_flip = np.clip(self.transfer(x, t, self.max_iter), 0.0, 1.0)
        do_flip = self.rng.random(x.shape) < p_flip
        return np.where(do_flip, 1 - prev_bit, prev_bit).astype(int)

    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:  # pragma: no cover - abstract-ish
        raise NotImplementedError


class BinaryPSO(_BinaryOptimizer):
    """Binary PSO kinh điển (Kennedy & Eberhart, 1997) với transfer V-shaped.

    Vận tốc liên tục, vị trí là binary mask; pbest/gbest là mask nhị phân. Vận
    tốc cập nhật theo hiệu (pbest − x) và (gbest − x), rồi binarize qua transfer.
    Đây là dạng BPSO chuẩn — KHÁC RG-SCSO chỉ ở: (a) cơ chế di chuyển PSO và
    (b) transfer KHÔNG dùng relevance.
    """

    def __init__(self, *args, w_max: float = 0.9, w_min: float = 0.4,
                 c1: float = 2.0, c2: float = 2.0, v_max: float = 6.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.v_max = v_max

    def _run(self) -> dict:
        # Khởi tạo: vị trí liên tục -> mask theo dấu; vận tốc = 0.
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
    """Binary GWO (Emary et al., 2016) với transfer V-shaped.

    Ba sói dẫn đầu α/β/δ (theo fitness của mask) giữ vị trí LIÊN TỤC; mỗi sói cập
    nhật vị trí liên tục theo công thức GWO gốc (hệ số a giảm 2 → 0), rồi binarize
    qua transfer để ra mask. KHÁC RG-SCSO chỉ ở: (a) cơ chế GWO và (b) transfer
    không relevance.
    """

    def _gwo_move(self, a: float, pos_i: np.ndarray,
                  alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """1 bước cập nhật vị trí liên tục theo 3 sói dẫn đầu (GWO gốc)."""
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

                # Cập nhật 3 sói dẫn đầu (giữ cả vị trí liên tục lẫn fitness).
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
