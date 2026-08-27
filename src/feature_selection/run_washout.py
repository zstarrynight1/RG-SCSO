"""Experiment #9 — Washout Sensitivity (diagnostic / mechanism figure for §III-B).

Tests the falsifiable prediction of the §III-B quantization argument:
"the washout rate rises as the search settles, and it depends on the transfer
function." Measured on REAL SCSO feature-selection trajectories (standard
S-shaped binary SCSO — the exact config that produced the Phase-3 null result),
NOT on synthetic positions.

For every iteration t of a genuine SCSO-FS run we take the population's continuous
positions {x_j} and probe each coordinate with a fixed-magnitude "continuous
enhancement" delta (|delta| = EPS, random sign — a stand-in for a Levy/DE step).
We then quantify how much of that enhancement survives the binarization transfer T,
under three complementary definitions so the DATA decides which cleanly supports
the claim (spec 8.1/4.2 — report honestly, do not tune):

    (1) decision washout  W_dec = 1 - P[ bit(x+delta) != bit(x) ]
        S-shaped bit:  x >= 0            (sigmoid >= 0.5)
        V-shaped flip: |x| >= atanh(0.5) (|tanh x| >= 0.5)
    (2) transmission      |T(x+delta) - T(x)|      (the Delta_j of the Proposition)
    (3) transfer slope    |T'(x)|                   (the ||T'|| leverage factor)

Aggregated over seeds -> washout vs iteration AND washout vs dimension, per transfer.

Single source of truth: writes experiments/results_washout/washout_raw.csv
(per dataset/seed/iteration) + washout_by_dim.csv (per dataset/transfer summary).
NO number is hand-typed; the figure script reads only these CSVs.

Run:  .venv/bin/python -m src.feature_selection.run_washout
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
from src.algorithms.scso import SCSO
from src.feature_selection.fitness import make_fitness_function
from src.feature_selection.run_feature_selection import _load_dataset

OUT_DIR = os.path.join("experiments", "results_washout")
# datasets spanning a range of dimensionality (small n -> keeps KNN-CV cheap)
DATASETS = ["HeartDisease", "Zoo", "Lymphography", "Parkinsons", "WDBC", "Sonar"]
SEEDS = [RANDOM_SEED_BASE + i for i in range(3)]
EPS = 0.10          # magnitude of the continuous-enhancement probe in [-1, 1] space
N_PROBE = 128       # Monte-Carlo probe draws per iteration snapshot
LB, UB = -1.0, 1.0
V_BOUND = float(np.arctanh(0.5))   # |tanh x| >= 0.5  <=>  |x| >= 0.5493


class TracingSCSO(SCSO):
    """SCSO that records the continuous population each iteration (faithful move)."""

    def _run(self) -> dict:
        population = self._init_population()
        fitness = self._evaluate_population(population)
        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        self.trajectory = []
        for t in range(self.max_iter):
            rG = self._sensitivity_range(t)
            self._scso_move_step(rG, population, fitness, best_solution)
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_solution = population[gen_best_idx].copy()
            self.trajectory.append(population.copy())
        return {"best_solution": best_solution, "best_fitness": best_fitness,
                "convergence_curve": []}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def washout_snapshot(pos: np.ndarray, rng: np.random.Generator) -> dict:
    """All washout statistics for one (pop x d) position snapshot."""
    x = pos.ravel()
    # decision washout (Monte-Carlo over fixed-magnitude, random-sign probes)
    dec_change_S = 0.0
    dec_change_V = 0.0
    for _ in range(N_PROBE):
        delta = EPS * rng.choice([-1.0, 1.0], size=x.shape)
        xp = x + delta
        dec_change_S += np.mean((x >= 0.0) != (xp >= 0.0))
        dec_change_V += np.mean((np.abs(x) >= V_BOUND) != (np.abs(xp) >= V_BOUND))
    resp_S = dec_change_S / N_PROBE
    resp_V = dec_change_V / N_PROBE
    # transmission |T(x+d)-T(x)| and slope |T'(x)| (analytic, sign-averaged)
    xp_pos, xp_neg = x + EPS, x - EPS
    trans_S = 0.5 * (np.abs(_sigmoid(xp_pos) - _sigmoid(x))
                     + np.abs(_sigmoid(xp_neg) - _sigmoid(x))).mean()
    v = lambda z: np.abs(np.tanh(z))
    trans_V = 0.5 * (np.abs(v(xp_pos) - v(x)) + np.abs(v(xp_neg) - v(x))).mean()
    s = _sigmoid(x)
    slope_S = np.mean(s * (1.0 - s))                 # sigma'(x)
    slope_V = np.mean(1.0 - np.tanh(x) ** 2)         # |tanh|' = sech^2
    return {"washout_dec_S": 1.0 - resp_S, "washout_dec_V": 1.0 - resp_V,
            "trans_S": trans_S, "trans_V": trans_V,
            "slope_S": slope_S, "slope_V": slope_V,
            "mean_abs_pos": float(np.abs(x).mean())}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in DATASETS:
        X, y = _load_dataset(ds)
        d = X.shape[1]
        for seed in SEEDS:
            obj = make_fitness_function(X, y, seed=seed)
            opt = TracingSCSO(obj_func=obj, dim=d, lb=LB, ub=UB,
                              pop_size=POPULATION_SIZE, max_iter=MAX_ITERATION, seed=seed)
            opt.optimize()
            probe_rng = np.random.default_rng(seed + 10_000)
            for t, pos in enumerate(opt.trajectory):
                st = washout_snapshot(pos, probe_rng)
                st.update({"dataset": ds, "dim": d, "seed": seed, "iteration": t})
                rows.append(st)
            print(f"  done {ds} (d={d}) seed={seed}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "washout_raw.csv"), index=False)

    # per (dataset, dim) mean over seeds & iterations -> washout vs dimension
    by_dim = (df.groupby(["dataset", "dim"])
                .agg(washout_dec_S=("washout_dec_S", "mean"),
                     washout_dec_V=("washout_dec_V", "mean"),
                     slope_S=("slope_S", "mean"), slope_V=("slope_V", "mean"))
                .reset_index().sort_values("dim"))
    by_dim.to_csv(os.path.join(OUT_DIR, "washout_by_dim.csv"), index=False)

    # quick console verdict: does decision-washout rise early -> late?
    print("\n=== washout(early 10%) vs (late 10%) of search, mean over datasets/seeds ===")
    n = df["iteration"].max() + 1
    early = df[df.iteration < 0.1 * n]
    late = df[df.iteration >= 0.9 * n]
    for tf in ("S", "V"):
        e, l = early[f"washout_dec_{tf}"].mean(), late[f"washout_dec_{tf}"].mean()
        print(f"  {tf}-shaped: early={e:.3f}  late={l:.3f}  rise={l - e:+.3f}")
    print("\nSaved:", os.path.join(OUT_DIR, "washout_raw.csv"),
          "and washout_by_dim.csv")


if __name__ == "__main__":
    main()
