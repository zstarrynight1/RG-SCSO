
from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.rg_scso import RGSCSO


class RGSCSOSignalPosition(RGSCSO):

    def __init__(
        self,
        *args,
        injection: str = "transfer",
        objective_penalty_weight: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if injection not in ("transfer", "init", "objective"):
            raise ValueError(f"injection không hợp lệ: {injection}")
        self.injection = injection
        self.objective_penalty_weight = objective_penalty_weight
        if injection in ("init", "objective"):
                                                                           
                                                                           
            self.use_rms = False

    def _init_population(self) -> np.ndarray:
        pos = super()._init_population()                                         
        if self.injection == "init":
            rho = self.relevance.rho_static
            preferred_sign = np.where(rho > 0.5, 1.0, -1.0)
            magnitude = np.abs(pos)                                     
            pos = preferred_sign[None, :] * magnitude
        return pos

    def _eval(self, mask: np.ndarray) -> float:
        base = super()._eval(mask)
        if self.injection == "objective" and self.objective_penalty_weight > 0:
            rho = self.relevance.rho_static
            preferred = (rho > 0.5).astype(float)
            selected = mask.astype(float)
                                                                           
            disagreement = float(np.mean(np.abs(selected - preferred)))
            return base + self.objective_penalty_weight * disagreement
        return base
