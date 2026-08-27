"""RG-SCSO — biến thể control NFE-matched cho UMR (RG-SCSO_MASTER_FINAL_
COMPLETE.md, mục 10 "NFE control" / 5 most-important-items #2).

CÂU HỎI: lợi ích của UMR (C3, memetic refinement trên K feature "biên" gần τ
nhất) đến từ việc TARGET đúng feature không chắc chắn, hay đơn thuần đến từ
việc có THÊM K lần đánh giá cục bộ ở bất kỳ đâu (extra search effort)? Ablation
hiện có (-UMR = tắt hẳn C3) không tách được 2 khả năng này — cả relevance-
targeting LẪN NFE bổ sung đều bị loại bỏ cùng lúc.

KHÔNG sửa rg_scso.py gốc — chỉ override `_memetic_refine`: thay vì chọn K
feature gần τ nhất (uncertainty-targeted), chọn K feature NGẪU NHIÊN (uniform,
qua self.rng — cùng generator, giữ reproducibility). Mọi thứ khác (ngân sách
NFE, umr_every, greedy-accept-if-better) giữ NGUYÊN Y HỆT bản gốc -> đúng
nghĩa "NFE-matched": tốn CHÍNH XÁC cùng số lần eval, chỉ khác chỗ đặt.

Nếu UMR (targeted) thắng RandomProbe với biên độ tương đương cách nó thắng
-UMR (không refine gì cả), bằng chứng cho thấy TARGETING mới là thứ tạo giá
trị, không phải chỉ "có thêm search effort". Nếu RandomProbe cho kết quả gần
ngang UMR targeted, đó là một null result thật cần công bố trung thực, không
phải kết quả cần né tránh — đúng nguyên tắc xuyên suốt dự án này.
"""

from __future__ import annotations

import numpy as np

from src.algorithms.rg_scso import RGSCSO


class RGSCSORandomProbe(RGSCSO):
    """RGSCSO với UMR (C3) thay bằng K feature ngẫu nhiên thay vì K feature
    'biên' gần τ nhất — NFE-matched control, xem docstring module."""

    def _memetic_refine(
        self, best_bits: np.ndarray, best_fit: float
    ) -> tuple[np.ndarray, float]:
        """Bản NFE-matched của C3: K feature NGẪU NHIÊN (không dùng ρ/τ),
        cùng ngân sách NFE, cùng logic greedy-accept-if-better như bản gốc."""
        d = best_bits.shape[0]
        candidates = self.rng.choice(d, size=min(self.umr_k, d), replace=False)
        bits = best_bits.copy()
        fit = best_fit
        for j in candidates:
            if self._nfe >= self.max_nfe:  # tôn trọng ngân sách NFE, giống hệt bản gốc
                break
            trial = bits.copy()
            trial[j] = 1 - trial[j]
            trial_fit = self._eval(trial)
            if trial_fit < fit:
                bits, fit = trial, trial_fit
        return bits, fit
