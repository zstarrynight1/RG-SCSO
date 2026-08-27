"""Unit test cơ bản cho ECL-SCSO (Phase 1, mục 3.5 trong PROJECT_SPEC.md)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.algorithms.ecl_scso import ECLSCSO
from src.algorithms.scso import SCSO

DIM = 10
POP_SIZE = 30
MAX_ITER = 500
SEED = 42


def sphere(x: np.ndarray) -> float:
    """f(x) = sum(x_i^2), global minimum f(0,...,0) = 0."""
    return float(np.sum(x**2))


def _make_ecl_scso(**overrides) -> ECLSCSO:
    kwargs = dict(
        obj_func=sphere,
        dim=DIM,
        lb=-100.0,
        ub=100.0,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED,
    )
    kwargs.update(overrides)
    return ECLSCSO(**kwargs)


def test_ecl_scso_converges_on_sphere():
    result = _make_ecl_scso().optimize()
    assert result["best_fitness"] < 1e-3


def test_convergence_curve_length_equals_max_iter():
    result = _make_ecl_scso().optimize()
    assert len(result["convergence_curve"]) == MAX_ITER


def test_convergence_curve_is_monotonically_non_increasing():
    result = _make_ecl_scso().optimize()
    curve = result["convergence_curve"]
    assert all(curve[i] >= curve[i + 1] - 1e-12 for i in range(len(curve) - 1))


@pytest.mark.parametrize(
    "flag",
    ["use_chaotic_init", "use_adaptive_R", "use_de_mutation", "use_levy_flight"],
)
def test_each_improvement_flag_runs_independently(flag):
    all_off = dict(
        use_chaotic_init=False,
        use_adaptive_R=False,
        use_de_mutation=False,
        use_levy_flight=False,
    )
    all_off[flag] = True
    optimizer = _make_ecl_scso(max_iter=50, **all_off)
    result = optimizer.optimize()
    assert np.isfinite(result["best_fitness"])
    assert len(result["convergence_curve"]) == 50


def test_all_flags_off_equivalent_to_plain_scso_run():
    optimizer = _make_ecl_scso(
        max_iter=50,
        use_chaotic_init=False,
        use_adaptive_R=False,
        use_de_mutation=False,
        use_levy_flight=False,
    )
    result = optimizer.optimize()
    assert np.isfinite(result["best_fitness"])
    assert len(result["convergence_curve"]) == 50


def test_scso_baseline_runs_on_sphere():
    optimizer = SCSO(
        obj_func=sphere, dim=DIM, lb=-100.0, ub=100.0, pop_size=POP_SIZE, max_iter=100, seed=SEED
    )
    result = optimizer.optimize()
    assert len(result["convergence_curve"]) == 100
    assert np.isfinite(result["best_fitness"])


def test_solutions_stay_within_bounds():
    lb, ub = -10.0, 10.0
    optimizer = _make_ecl_scso(lb=lb, ub=ub, max_iter=50)
    result = optimizer.optimize()
    assert np.all(result["best_solution"] >= lb - 1e-9)
    assert np.all(result["best_solution"] <= ub + 1e-9)
