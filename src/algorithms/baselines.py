"""Wrapper gọi 9 thuật toán baseline qua thư viện `mealpy` (v3.x).

Mọi wrapper trả về CÙNG format dict như `BaseOptimizer.optimize()`:
    {"best_solution": np.ndarray, "best_fitness": float,
     "convergence_curve": list[float], "runtime": float}
để pipeline downstream (benchmark, feature selection) xử lý đồng nhất,
không cần biết baseline nào đang chạy.

LƯU Ý ĐẶT TÊN TRONG MEALPY (dễ nhầm):
    - `mealpy.swarm_based.COA`      = Coyote Optimization Algorithm (KHÔNG
      phải Coati).
    - `mealpy.swarm_based.CoatiOA`  = Coati Optimization Algorithm — đây mới
      là "COA (Coati)" mà spec yêu cầu, nên dùng `CoatiOA.OriginalCoatiOA`.
    - "AOA" trong spec được hiểu là Arithmetic Optimization Algorithm
      (Abualigah et al., 2021) — `mealpy.math_based.AOA.OriginalAOA`. Đây là
      nghĩa phổ biến nhất của "AOA" trong tài liệu feature-selection, KHÔNG
      phải Archimedes Optimization Algorithm.

Đã xác minh (2026-06-21) cả 9 thuật toán đều có sẵn trong mealpy==3.0.3,
class "Original*", API thống nhất: `model.solve(problem_dict, seed=...)`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from mealpy import FloatVar
from mealpy.evolutionary_based.GA import OriginalGA
from mealpy.math_based.AOA import OriginalAOA
from mealpy.physics_based.RIME import OriginalRIME
from mealpy.swarm_based.CoatiOA import OriginalCoatiOA
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.OOA import OriginalOOA
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.SSA import OriginalSSA
from mealpy.swarm_based.WOA import OriginalWOA

# algorithm_name -> mealpy "Original*" class
_MEALPY_ALGORITHMS = {
    "GA": OriginalGA,
    "PSO": OriginalPSO,
    "GWO": OriginalGWO,
    "WOA": OriginalWOA,
    "HHO": OriginalHHO,
    "SSA": OriginalSSA,
    "AOA": OriginalAOA,
    "COA": OriginalCoatiOA,  # Coati Optimization Algorithm (xem ghi chú ở trên)
    "OOA": OriginalOOA,
    # SOTA gần đây (2023) làm mốc "recent" cho so sánh Q1 — Rime Optimization
    # Algorithm (Su et al., Neurocomputing 2023). Chạy CÙNG pipeline binarize
    # (sigmoid + ngưỡng) như các baseline khác để công bằng tuyệt đối.
    "RIME": OriginalRIME,
}


def run_mealpy_baseline(
    algorithm_name: str,
    obj_func: Callable[[np.ndarray], float],
    dim: int,
    lb,
    ub,
    pop_size: int,
    max_iter: int,
    seed: int,
) -> dict:
    """Chạy 1 baseline mealpy, trả về dict cùng format với `BaseOptimizer.optimize()`.

    Args:
        algorithm_name: 1 trong các key của `_MEALPY_ALGORITHMS`
            (GA, PSO, GWO, WOA, HHO, SSA, AOA, COA, OOA).
    """
    if algorithm_name not in _MEALPY_ALGORITHMS:
        raise ValueError(
            f"Không hỗ trợ baseline '{algorithm_name}'. "
            f"Các baseline hợp lệ: {sorted(_MEALPY_ALGORITHMS)}"
        )

    lb_vec = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_vec = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    problem = {
        "bounds": FloatVar(lb=lb_vec.tolist(), ub=ub_vec.tolist()),
        "minmax": "min",
        "obj_func": obj_func,
        "log_to": None,  # tắt log console của mealpy, pipeline tự quản lý progress bar
    }

    model_cls = _MEALPY_ALGORITHMS[algorithm_name]
    model = model_cls(epoch=max_iter, pop_size=pop_size)

    start = time.perf_counter()
    g_best = model.solve(problem, seed=seed)
    runtime = time.perf_counter() - start

    return {
        "best_solution": np.asarray(g_best.solution, dtype=float),
        "best_fitness": float(g_best.target.fitness),
        "convergence_curve": list(model.history.list_global_best_fit),
        "runtime": runtime,
    }
