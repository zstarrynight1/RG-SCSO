"""Baseline SCSO-FS CÙNG HỌ (reimplement) — trả lời rev #2: bài tuyên bố "gap"
ngay trong dòng SCSO-based feature selection, nên PHẢI so trực tiếp với chính các
biến thể SCSO-FS đã công bố, không chỉ với SCSO gốc + các họ optimizer khác.

HAI CẤU HÌNH đại diện công thức chuẩn của dòng này:
    bSCSO-S   : SCSO move (Eq. 4-8) + transfer S-shaped (gán TUYỆT ĐỐI:
                bit = 1 nếu rand < σ(x)) — công thức binarize phổ biến nhất của
                các bScSO/bSCSO sơ khởi.
    bSCSO-OBL : SCSO move + transfer V-shaped (lật TƯƠNG ĐỐI) + opposition-based
                learning ở khởi tạo (đánh giá N nghiệm gốc + N nghiệm đối, giữ N
                tốt nhất) — đại diện nhánh "improved SCSO-FS" (OBL là cải tiến
                được dùng lại nhiều nhất trong nhóm 2022-2024).

KHÁC BIỆT DUY NHẤT với RG-SCSO: các baseline này dùng transfer feature-agnostic
(không có trường relevance per-feature) — đúng cái RG-SCSO thêm vào. Nếu RG-SCSO
vẫn parsimonious hơn/không kém accuracy, đóng góp lõi được cô lập so với CHÍNH họ
SCSO-FS.

LƯU Ý SOURCING (BẮT BUỘC đọc trước khi đưa vào Methodology — giống caveat scso.py):
    Đây là REIMPLEMENTATION theo mô tả/pseudocode chuẩn của dòng SCSO-FS, KHÔNG
    phải bản chính chủ của một bài cụ thể. SCSO move tái dùng đúng Eq. (4)-(8) đã
    đối chiếu trong scso.py. Transfer S/V theo Mirjalili & Lewis (2013); OBL theo
    Tizhoosh (2005). TRƯỚC KHI công bố phải:
      1. Đối chiếu tham số (kích thước quần thể, lịch trình, biến thể OBL) với
         đúng bài được trích [12]/[13]/[14];
      2. Khai báo minh bạch trong bài: "reimplemented from [ref] based on the
         published pseudocode; parameters as reported" — reviewer chấp nhận nếu
         nêu rõ, KHÔNG được ngầm cho là bản chính chủ.
    Giao thức đánh giá (eval_mask, ngân sách NFE, biên, seed) KHỚP tuyệt đối với
    RG-SCSO và các baseline khác để so sánh công bằng.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.binary_baselines import _BinaryOptimizer
from src.feature_selection.transfer_function import sigmoid, v_shaped

# Transfer "kind" -> hàm p (xác suất), dùng để thoả interface _BinaryOptimizer;
# binarize thực tế do BinarySCSO tự xử (S = gán tuyệt đối, V = lật tương đối).
_TRANSFER_P = {
    "s": lambda x, t, mi: sigmoid(x),
    "v": lambda x, t, mi: v_shaped(x),
}


class BinarySCSO(_BinaryOptimizer):
    """SCSO nhị phân cho feature selection (reimplementation dòng SCSO-FS).

    Tái dùng ĐÚNG luật di chuyển SCSO Eq. (4)-(8) (sensitivity range rG giảm
    tuyến tính; pha explore |R|>1, exploit |R|<=1), binarize bằng transfer
    S-shaped (tuyệt đối) hoặc V-shaped (tương đối), tuỳ chọn opposition-based
    learning ở khởi tạo. KHÔNG có relevance per-feature (đó là phần RG-SCSO thêm).
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
        S_M: float = 2.0,
        transfer_kind: str = "s",
        use_obl: bool = False,
        max_nfe: int | None = None,
    ) -> None:
        if transfer_kind not in _TRANSFER_P:
            raise ValueError(f"transfer_kind phải là 's' hoặc 'v', nhận '{transfer_kind}'.")
        super().__init__(
            obj_func, dim, lb, ub, pop_size, max_iter, seed,
            eval_mask=eval_mask, transfer=_TRANSFER_P[transfer_kind], max_nfe=max_nfe,
        )
        self.S_M = S_M
        self.transfer_kind = transfer_kind
        self.use_obl = use_obl

    # ------------------------------------------------------------------ helpers
    def _binarize_pos(self, x: np.ndarray, prev_bit: np.ndarray) -> np.ndarray:
        """S-shaped: gán tuyệt đối bit=1 nếu rand<σ(x); V-shaped: lật tương đối."""
        if self.transfer_kind == "s":
            return (self.rng.random(x.shape) < sigmoid(x)).astype(int)
        p_flip = np.clip(v_shaped(x), 0.0, 1.0)
        do_flip = self.rng.random(x.shape) < p_flip
        return np.where(do_flip, 1 - prev_bit, prev_bit).astype(int)

    def _sensitivity_range(self, t: int) -> float:
        """rG(t) — Eq. (3): giảm tuyến tính S_M -> 0 (đồng bộ scso.py)."""
        return self.S_M - self.S_M * t / self.max_iter

    def _scso_step(self, rG: float, pos_i: np.ndarray, best_pos: np.ndarray,
                   pop: np.ndarray) -> np.ndarray:
        """1 bước di chuyển liên tục theo Eq. (4)-(8) (vô hướng mỗi cá thể)."""
        R = 2.0 * rG * self.rng.random() - rG          # Eq. (4)
        r = rG * self.rng.random()                     # Eq. (5)
        if abs(R) > 1.0:                               # explore — Eq. (6)
            p_rand = pop[self.rng.integers(self.pop_size)]
            new_pos = r * (p_rand - self.rng.random() * pos_i)
        else:                                          # exploit — Eq. (7)-(8)
            pr = self.rng.random() * best_pos - pos_i
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            new_pos = best_pos - r * pr * np.cos(theta)
        return self._clip(new_pos)

    def _obl_init(self, pos: np.ndarray, bits: np.ndarray,
                  fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Opposition-based init: đánh giá N nghiệm đối (x_opp = lb+ub−x), ghép
        2N, giữ N tốt nhất. Mọi eval đối đều tính vào ngân sách NFE (công bằng).
        """
        opp_pos = self.lb + self.ub - pos  # self.lb/ub luôn là array shape (dim,)
        opp_bits = np.array([self._binarize_pos(opp_pos[i], bits[i])
                             for i in range(self.pop_size)])
        opp_fit = np.array([self._eval(b) for b in opp_bits], dtype=float)
        all_pos = np.vstack([pos, opp_pos])
        all_bits = np.vstack([bits, opp_bits])
        all_fit = np.concatenate([fitness, opp_fit])
        keep = np.argsort(all_fit)[: self.pop_size]
        return all_pos[keep], all_bits[keep], all_fit[keep]

    # ------------------------------------------------------------------ run loop
    def _run(self) -> dict:
        pos = self._init_population()
        bits = np.array([self._binarize_pos(pos[i], (pos[i] > 0).astype(int))
                         for i in range(self.pop_size)])
        fitness = np.array([self._eval(b) for b in bits], dtype=float)

        if self.use_obl and self._nfe < self.max_nfe:
            pos, bits, fitness = self._obl_init(pos, bits, fitness)

        best_idx = int(np.argmin(fitness))
        best_pos = pos[best_idx].copy()
        best_bits = bits[best_idx].copy()
        best_fit = float(fitness[best_idx])
        convergence_curve = []

        for t in range(self.max_iter):
            if self._nfe >= self.max_nfe:
                break
            rG = self._sensitivity_range(t)
            for i in range(self.pop_size):
                if self._nfe >= self.max_nfe:
                    break
                new_pos = self._scso_step(rG, pos[i], best_pos, pos)
                new_bits = self._binarize_pos(new_pos, bits[i])
                new_fit = self._eval(new_bits)
                pos[i], bits[i], fitness[i] = new_pos, new_bits, new_fit

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_fit:
                best_fit = float(fitness[gen_best])
                best_pos = pos[gen_best].copy()
                best_bits = bits[gen_best].copy()
            convergence_curve.append(best_fit)

        if convergence_curve:
            convergence_curve.extend(
                [convergence_curve[-1]] * (self.max_iter - len(convergence_curve))
            )
        return {
            "best_solution": np.where(best_bits > 0, 1.0, -1.0),
            "best_mask": best_bits,
            "best_fitness": best_fit,
            "convergence_curve": convergence_curve,
        }
