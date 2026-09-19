
from __future__ import annotations

import numpy as np

from src.algorithms.rg_scso import RGSCSO


class RGSCSORandomProbe(RGSCSO):

    def _memetic_refine(
        self, best_bits: np.ndarray, best_fit: float
    ) -> tuple[np.ndarray, float]:
        d = best_bits.shape[0]
        candidates = self.rng.choice(d, size=min(self.umr_k, d), replace=False)
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
