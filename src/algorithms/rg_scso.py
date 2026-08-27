"""RG-SCSO — Relevance-Guided Sand Cat Optimizer cho binary feature selection.

ĐỘNG CƠ (vì sao thiết kế thế này):
    Phase 3 cho kết quả null trung thực — các cải tiến continuous của ECL-SCSO
    (chaotic init, adaptive R, DE, Lévy) KHÔNG truyền được sang feature selection
    vì bị `sigmoid + ngưỡng 0.5` lượng tử hóa mất. RG-SCSO giải quyết tận gốc:
    thay vì thêm toán tử continuous rồi ép nhị phân, nó là thuật toán
    BINARY-NATIVE, diễn giải lại chính cơ chế sinh học lõi của SCSO — "sensitivity
    range" (vùng nghe tần số thấp của mèo cát, co dần để chuyển explore->exploit)
    — thành ĐỘ NHẠY LẬT-BIT theo TỪNG feature, được dẫn hướng bởi một trường
    độ-liên-quan ρ học online.

BA THÀNH PHẦN (mỗi cái có cờ bật/tắt để ablation chứng minh load-bearing):
    C1 — Relevance-Modulated Sensitivity (use_rms): binarize bằng V-shaped của vị
         trí SCSO, điều biến ngưỡng lật theo ρ (`binarize_relevance`).
    C2 — Online Relevance Learning (use_orl): ρ = prior filter tĩnh + điểm học
         online (EMA credit-assignment từ độ thay đổi fitness).
    C3 — Uncertainty-Targeted Memetic Refinement (use_umr): mỗi vòng, local search
         lật thử K feature "biên" (ρ gần 0.5 nhất) trên nghiệm best.

Kế thừa SCSO để tái dùng ĐÚNG công thức di chuyển gốc Eq. (3)-(8); chỉ thay tầng
binarize + đánh giá (dùng `eval_mask` trên mask nhị phân thay cho `obj_func` trên
vị trí liên tục) và thêm bước memetic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.scso import SCSO
from src.feature_selection.relevance import RelevanceField, relevance_prior
from src.feature_selection.transfer_function import binarize_relevance, v_shaped


class RGSCSO(SCSO):
    """Relevance-Guided SCSO — binary feature selection wrapper optimizer."""

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
        use_orl: bool = False,  # C2 CẮT: ablation (R4) chứng minh không load-bearing
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
        """
        Args:
            obj_func: Giữ cho tương thích interface BaseOptimizer (KHÔNG dùng để
                chấm điểm — RG-SCSO chấm trực tiếp trên mask qua `eval_mask`).
            dim: Số feature (= chiều không gian tìm kiếm liên tục).
            lb, ub: Biên không gian liên tục (giống baseline: -1, 1).
            X, y: Dữ liệu, để tính prior relevance ρ_static.
            eval_mask: Callback nhận binary mask -> fitness (bọc
                `evaluate_binary_mask(...)['fitness']`). Fitness càng nhỏ càng tốt.
            use_rms/use_orl/use_umr: Cờ bật 3 thành phần (ablation). MẶC ĐỊNH
                use_orl=False: ablation R4 cho thấy Online Relevance Learning
                KHÔNG load-bearing (gỡ ra không giảm accuracy có ý nghĩa trên
                bất kỳ dataset nào) -> CẮT khỏi thuật toán cuối (spec 8.1/4.2,
                không giữ trang trí). RG-SCSO cuối = RMS (relevance tĩnh MI) + UMR.
                Ablation vẫn override tường minh để tái tạo cấu hình 3-thành-phần.
            gamma: Cường độ điều biến relevance ở C1 (0 = V-shaped thuần).
            umr_k: Số feature "biên" thử lật mỗi lần memetic (C3).
            umr_every: Chu kỳ (số vòng) giữa 2 lần memetic.
            ema_lambda, w_online, delta_scale: Tham số RelevanceField (C2).
            max_nfe: NGÂN SÁCH số lần đánh giá fitness (function evaluations).
                None -> pop_size × max_iter (KHỚP với baseline SCSO/mealpy để so
                sánh công bằng: memetic C3 tốn thêm eval nên phải giới hạn tổng
                NFE, nếu không RG-SCSO được "ngân sách nhiều hơn" — reviewer Q1
                sẽ bắt lỗi). Đếm MỌI lần gọi eval_mask (init + move + memetic).
            prior_method: Nguồn relevance prior ρ_static ("mi" mặc định = bài
                chính; "relieff"/"fisher" cho thí nghiệm robustness rev #6, để
                chứng minh lợi ích không phải artifact riêng của MI).
            tau: Ngưỡng τ phân tách bit "được ưu tiên" trong RMS (0.5 = mặc
                định gốc, tái tạo đúng 100% mọi kết quả đã có). Dùng cho
                threshold-sensitivity study (Q1 review), sweep τ∈{0.4,0.5,0.6}
                — xem `binarize_relevance` để biết cách tổng quát hóa `strength`
                giữ đúng tính chất bị chặn [0,1] với mọi τ. `_memetic_refine`
                (C3) cũng chọn K feature "biên" theo |ρ−τ| gần 0 nhất, nhất
                quán với ngưỡng quyết định thật của RMS, không còn cố định 0.5.
        """
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
        # Instrumentation (rev — phòng thủ tử huyệt E&E/đóng băng bit): mỗi vòng ghi
        # đa dạng nhị phân + tỉ lệ bit đóng băng của quần thể. Chỉ bật khi cần đo.
        self.record_history = record_history
        self.history: list[dict] = []

        rho_static = relevance_prior(X, y, seed, method=prior_method)
        self.relevance = RelevanceField(
            rho_static, ema_lambda=ema_lambda, w_online=w_online, delta_scale=delta_scale
        )

    # ------------------------------------------------------------------ helpers
    def _eval(self, mask: np.ndarray) -> float:
        """Chấm điểm 1 mask và tăng bộ đếm NFE (mọi eval đều tính vào ngân sách)."""
        self._nfe += 1
        return self.eval_mask(mask)

    def _rho(self) -> np.ndarray:
        """ρ hiện tại: hợp nhất nếu bật ORL, prior tĩnh nếu tắt."""
        return self.relevance.combined() if self.use_orl else self.relevance.rho_static

    def _binarize(self, x: np.ndarray, prev_bit: np.ndarray) -> np.ndarray:
        """Nhị phân hóa vị trí liên tục -> mask, dùng C1 nếu bật, V-shaped thuần
        nếu tắt (ablation NoRMS)."""
        if self.use_rms:
            return binarize_relevance(
                x, prev_bit, self._rho(), self.rng, self.gamma, self.tau
            )
        # V-shaped thuần: lật bit theo xác suất |tanh(x)|, không dùng relevance.
        do_flip = self.rng.random(x.shape) < v_shaped(x)
        return np.where(do_flip, 1 - prev_bit, prev_bit).astype(int)

    def _memetic_refine(
        self, best_bits: np.ndarray, best_fit: float
    ) -> tuple[np.ndarray, float]:
        """C3 — greedy bit-flip trên K feature 'biên' (ρ gần τ nhất) của best."""
        rho = self._rho()
        uncertainty = np.abs(rho - self.tau)  # nhỏ = biên/không chắc, đúng ngưỡng quyết định thật của RMS
        candidates = np.argsort(uncertainty)[: self.umr_k]
        bits = best_bits.copy()
        fit = best_fit
        for j in candidates:
            if self._nfe >= self.max_nfe:  # tôn trọng ngân sách NFE
                break
            trial = bits.copy()
            trial[j] = 1 - trial[j]
            trial_fit = self._eval(trial)
            if trial_fit < fit:
                bits, fit = trial, trial_fit
        return bits, fit

    # ------------------------------------------------------------------ run loop
    def optimize(self) -> dict:
        return self._timed(self._run)

    def _run(self) -> dict:
        # Quần thể liên tục + mask nhị phân song song (V-shaped lật bit TƯƠNG ĐỐI
        # so với bit hiện tại nên phải giữ trạng thái bit).
        pos = self._init_population()
        bits = (pos > 0.0).astype(int)  # khởi tạo mask theo dấu vị trí
        fitness = np.array([self._eval(b) for b in bits], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_bits = bits[best_idx].copy()
        best_pos = pos[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        convergence_curve = []

        # Dừng theo max_iter HOẶC hết ngân sách NFE (whichever first) để công bằng
        # với baseline: tổng số eval của RG-SCSO không vượt pop_size × max_iter.
        for t in range(self.max_iter):
            if self._nfe >= self.max_nfe:
                break
            rG = self._sensitivity_range(t)  # Eq. (3) — tái dùng SCSO
            for i in range(self.pop_size):
                if self._nfe >= self.max_nfe:
                    break
                # --- Bước di chuyển SCSO Eq. (4)-(8) (đồng bộ với SCSO gốc) ---
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

                # --- C1: binarize dẫn hướng relevance ---
                new_bits = self._binarize(new_pos, bits[i])
                new_fit = self._eval(new_bits)

                # --- C2: học online từ độ thay đổi fitness ---
                if self.use_orl:
                    self.relevance.update(new_bits, fitness[i] - new_fit)

                # SCSO ghi đè vị trí mỗi vòng (non-elitist theo cá thể).
                pos[i], bits[i], fitness[i] = new_pos, new_bits, new_fit

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_bits = bits[gen_best_idx].copy()
                best_pos = pos[gen_best_idx].copy()

            # --- C3: memetic refinement trên nghiệm best ---
            if self.use_umr and t % self.umr_every == 0:
                best_bits, best_fitness = self._memetic_refine(best_bits, best_fitness)

            convergence_curve.append(best_fitness)

            if self.record_history:
                # p_j = tỉ lệ cá thể chọn feature j; đa dạng nhị phân = kỳ vọng
                # Hamming chuẩn hóa mean_j 2·p_j·(1−p_j) ∈ [0,0.5]; bit "đóng băng"
                # = mọi cá thể đồng thuận (p_j ∈ {0,1}).
                p = bits.mean(axis=0)
                self.history.append({
                    "iter": t,
                    "diversity": float(np.mean(2.0 * p * (1.0 - p))),
                    "frozen_frac": float(np.mean((p == 0.0) | (p == 1.0))),
                    "mean_subset_size": float(bits.sum(axis=1).mean()),
                })

        # Pad đường hội tụ tới đúng max_iter (giữ giá trị best cuối) để độ dài
        # đồng nhất với các thuật toán khác khi vẽ/ghép biểu đồ.
        if convergence_curve:
            convergence_curve.extend(
                [convergence_curve[-1]] * (self.max_iter - len(convergence_curve))
            )

        return {
            "best_solution": best_pos,
            "best_mask": best_bits,
            "best_fitness": best_fitness,
            "convergence_curve": convergence_curve,
        }
