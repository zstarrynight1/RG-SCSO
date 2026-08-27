"""RG-SCSO — biến thể cô lập VỊ TRÍ tiêm tín hiệu relevance (Q1 review
Priority 2 / W7): "đưa MI vào binarization operator có thực sự quan trọng
hơn đưa MI vào initialization/objective hay không?"

KHÔNG sửa rg_scso.py gốc — mọi kết quả bài chính giữ nguyên, không rủi ro.
Đây là bản dẫn xuất RIÊNG chỉ dùng cho ablation cô lập vị trí tín hiệu, tái
dùng 100% vòng lặp tìm kiếm SCSO gốc của RGSCSO, chỉ override 2 điểm:

    injection="init"      : quần thể khởi tạo lệch theo dấu ưu tiên của ρ_j
                             (feature liên quan -> khởi tạo dương; nhiễu ->
                             âm), rồi search tiếp diễn HOÀN TOÀN
                             relevance-agnostic (RMS tắt, V-shaped thuần).
    injection="objective"  : eval_mask được cộng thêm một số hạng phạt/thưởng
                             tuyến tính theo mức "đồng thuận" giữa mask được
                             chọn và ρ (chọn feature ρ>0.5, bỏ feature
                             ρ<0.5), RMS tắt trong suốt binarization.
    injection="transfer"   : hành vi RG-SCSO gốc (RMS bật) — dùng để tái tạo
                             bước #4/#5 trong chuỗi 5 bước mà KHÔNG cần
                             import 2 class khác nhau trong harness.

5 bước ablation đầy đủ (harness run_fs_signal_position.py):
    1. injection="transfer", use_rms=False, use_umr=False  (= NoImprovement,
       đã có sẵn qua RGSCSO gốc, không cần class này)
    2. injection="init",     use_rms=False, use_umr=False
    3. injection="objective",use_rms=False, use_umr=False
    4. injection="transfer", use_rms=True,  use_umr=False  (= NoUMR, RGSCSO gốc)
    5. injection="transfer", use_rms=True,  use_umr=True   (= Full, RGSCSO gốc)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.rg_scso import RGSCSO


class RGSCSOSignalPosition(RGSCSO):
    """RGSCSO với tùy chọn tiêm tín hiệu relevance vào init/objective thay vì
    (hoặc cùng với) binarization/RMS. Xem docstring module để biết 3 chế độ."""

    def __init__(
        self,
        *args,
        injection: str = "transfer",
        objective_penalty_weight: float = 0.05,
        **kwargs,
    ) -> None:
        """
        Args:
            injection: "transfer" (mặc định, hành vi RGSCSO gốc), "init"
                (MI-guided initialization, RMS tắt cưỡng bức), "objective"
                (MI-weighted objective penalty, RMS tắt cưỡng bức).
            objective_penalty_weight: Trọng số phạt/thưởng cho injection=
                "objective", cùng đơn vị với fitness (error-rate scale
                0.99·err + 0.01·|b|/d) — 0.05 đủ lớn để tạo áp lực chọn lọc
                rõ rệt mà không lấn át error term (Acc chiếm 0.99).
        """
        super().__init__(*args, **kwargs)
        if injection not in ("transfer", "init", "objective"):
            raise ValueError(f"injection không hợp lệ: {injection}")
        self.injection = injection
        self.objective_penalty_weight = objective_penalty_weight
        if injection in ("init", "objective"):
            # Đảm bảo RMS KHÔNG hoạt động — tín hiệu chỉ được phép vào đúng
            # MỘT điểm (init HOẶC objective), không rò rỉ qua binarization.
            self.use_rms = False

    def _init_population(self) -> np.ndarray:
        pos = super()._init_population()  # uniform [lb, ub], giữ đúng RNG stream
        if self.injection == "init":
            rho = self.relevance.rho_static
            preferred_sign = np.where(rho > 0.5, 1.0, -1.0)
            magnitude = np.abs(pos)  # giữ nguyên biên độ ngẫu nhiên gốc
            pos = preferred_sign[None, :] * magnitude
        return pos

    def _eval(self, mask: np.ndarray) -> float:
        base = super()._eval(mask)
        if self.injection == "objective" and self.objective_penalty_weight > 0:
            rho = self.relevance.rho_static
            preferred = (rho > 0.5).astype(float)
            selected = mask.astype(float)
            # disagreement ∈ [0,1]: tỉ lệ feature mask KHÔNG khớp ρ ưu tiên
            disagreement = float(np.mean(np.abs(selected - preferred)))
            return base + self.objective_penalty_weight * disagreement
        return base
