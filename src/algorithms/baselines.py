
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

                                            
_MEALPY_ALGORITHMS = {
    "GA": OriginalGA,
    "PSO": OriginalPSO,
    "GWO": OriginalGWO,
    "WOA": OriginalWOA,
    "HHO": OriginalHHO,
    "SSA": OriginalSSA,
    "AOA": OriginalAOA,
    "COA": OriginalCoatiOA,                                                     
    "OOA": OriginalOOA,


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
        "log_to": None,                                                                
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
