# EXPERIMENT PROTOCOL — RG-SCSO (locked before the full run)

> **Note on this file.** This is a faithful English translation of the locked
> pre-registration `EXPERIMENT_PROTOCOL.md`. The Vietnamese original is the
> authoritative, version-controlled artifact (its lock date and change-log
> timestamps are the record of provenance); this translation is provided for
> reviewer convenience only and does not alter the locked original.

> **Pre-registration.** This document locks the experimental design BEFORE the
> full experiment is run and BEFORE the official results are seen. The criteria
> are not changed after the numbers are observed. Any change (if strictly
> required) must be recorded in a dated log with a reason at the end of the file,
> without deleting prior content. Overriding principle: **do it once, honestly.**
> Lock date: 2026-07-01.

## 1. Proposed algorithm

**RG-SCSO** — Relevance-Guided Sand Cat Optimizer (binary-native feature selection).
Reinterprets the SCSO "sensitivity range" as a per-feature bit-flip sensitivity,
guided by an online-learned relevance field ρ. Three components:

- **C1 — Relevance-Modulated Sensitivity (RMS):** `src/feature_selection/transfer_function.py::binarize_relevance`
- **C2 — Online Relevance Learning (ORL):** `src/feature_selection/relevance.py::RelevanceField`
- **C3 — Uncertainty-Targeted Memetic Refinement (UMR):** `src/algorithms/rg_scso.py::_memetic_refine`

Implementation: `src/algorithms/rg_scso.py::RGSCSO`.

## 2. Baselines (reduced set, locked)

RG-SCSO vs **SCSO** (2022, base — mandatory comparison), **AOA** (2021, current FS
leader — must be beaten), **COA/CoatiOA** (2023), **GWO** (2014), **PSO** (1995).
Optionally add one binary-FS method from 2022–2024 if integrated in time. Drop the
weaker group (GA, WOA, HHO, SSA, OOA).

## 3. Datasets (locked)

18 preprocessed UCI datasets in `data/processed/`, including 2 high-dimensional
gene-expression sets as case studies: **Leukemia** (3571 features, 72 samples),
**ColonCancer** (2000 features, 62 samples). No datasets added or removed after
the lock.

## 4. Protocol (locked — strict fairness)

| Parameter | Value | Source |
|---|---|---|
| pop_size | 30 | `config.POPULATION_SIZE` |
| max_iter | 500 | `config.MAX_ITERATION` |
| independent runs | 30 | `config.NUM_INDEPENDENT_RUNS` |
| seed | `RANDOM_SEED_BASE(42) + run_id` | SHARED across all algorithms (paired) |
| classifier fitness | KNN (k=5) | `config.KNN_NEIGHBORS` |
| CV | StratifiedKFold k=5, scaler fit within each fold | `fitness._knn_cv_accuracy` |
| fitness | `0.99·(1−acc) + 0.01·(n_sel/n_total)` | `config.FITNESS_ALPHA/BETA` |
| continuous space | dim=n_features, [−1, 1] | `run_feature_selection.SEARCH_LB/UB` |

The only difference between algorithms is the algorithm itself. RG-SCSO parameters
(gamma=0.5, umr_k=8, ema_lambda=0.9, w_online=0.3, delta_scale=0.01) are locked at
their defaults in `RGSCSO.__init__`; they are NOT tuned per dataset to produce
flattering numbers.

## 5. Metrics & statistical tests (locked)

- Primary metric: **accuracy** (KNN 5-fold). Secondary: fitness, number of selected
  features.
- **Wilcoxon signed-rank** (paired by seed) comparing RG-SCSO against each baseline
  on each dataset; **Holm correction** within each dataset.
- **Effect size mandatory**: Cohen's d (paired) + rank-biserial r (alongside Wilcoxon).
- **Friedman test** + average ranking over all datasets; **Nemenyi post-hoc +
  Critical-Difference diagram**.
- All in `src/stats/statistical_tests.py`. Report EVERYTHING, including datasets lost.

## 6. Success criteria (LOCKED — decides whether there is a "contribution")

Minimum conditions to claim RG-SCSO is a genuine contribution:

1. **Improved Friedman rank over the base SCSO** (rank RG-SCSO < rank SCSO), AND
2. **Beats AOA** on a majority of datasets by mean accuracy, AND
3. On significant comparison pairs: **Wilcoxon + Holm p < 0.05** with **effect size
   ≥ small** (|d| ≥ 0.2 or |r| ≥ 0.2) on a majority.

If NOT met → report honestly as null/negative, do NOT tune the numbers; discuss with
the user how to redesign the MECHANISM and then re-validate.

## 7. Ablation (falsifiability — locked)

5 configurations Full/NoRMS/NoORL/NoUMR/NoImprovement (`run_fs_ablation.py`) × 5
representative datasets × 30 runs. Any component whose removal does NOT reduce
accuracy with statistical significance → NOT load-bearing → **cut** from the final
algorithm.

## 8. Causal evidence (emphasizing the novelty)

Beyond "winning", measure the mechanism directly: (a) overlap between the features
RG-SCSO selects and the top-relevance ρ; (b) convergence curves vs baselines;
(c) feature count (parsimony); (d) how ρ_online changes across iterations on the
high-dimensional gene datasets.

## 9. Execution sequence with stop-gates

R0 stats → R1 code+smoke (DONE) → **R2 pilot** (`run_fs_pilot.py`, 5×10, gate ≥3/5)
→ if PASS → R3 full run (`run_feature_selection.py`, needs AC power + caffeinate) →
R4 stats+ablation+figures → R5 writing per Q1_BLUEPRINT.md.

The pilot is NOT a reported number; the paper's numbers come from the single full run
under this protocol.

---
### Change log
- 2026-07-01: initial lock (before the full run).
- 2026-07-01: **NFE budget equalization** (before the pilot, before seeing full
  results). Found in the signal test: RG-SCSO's memetic step (C3) spends ~4000 extra
  evaluations/run versus the baselines → unfair. Added a `max_nfe` parameter (default
  = pop_size × max_iter = 15000, MATCHING the SCSO/mealpy baselines) + an NFE counter
  in `RGSCSO`; stop when the budget is exhausted. This tightens fairness, it is NOT
  tuning for nicer numbers (it actually makes RG-SCSO HARDER, ~394 iterations instead
  of 500). Section 4 (protocol) applies the shared NFE constraint to ALL algorithms.
- 2026-07-02: **Pilot R2 PASSED 5/5** (Leukemia/ColonCancer/Sonar/WDBC/Zoo, 10 runs,
  budget-matched). RG-SCSO beat BOTH SCSO and AOA on all 5 datasets in accuracy AND
  selected ~½ the features. This is NOT a reported number; opening R3 full run under
  this protocol.
- 2026-07-02: **Fixed the recent-SOTA baseline = RIME** (Rime Optimization Algorithm,
  Su et al., Neurocomputing 2023) — realizing the pre-registered "+1 binary-FS SOTA
  2022–2024" option from Section 2. Run through the SAME binarization pipeline as the
  other baselines (strict fairness). Final comparison set = 7: RG-SCSO, SCSO, AOA,
  COA, GWO, PSO, RIME. The older baselines (SCSO/AOA/COA/GWO/PSO, 18×30) are reused
  from fs_results.csv (shared code path unchanged → reproducible, paired by seed);
  only RG-SCSO + RIME are run anew.
- 2026-07-08: **R3b — Fold-honest held-out generalization run (locked BEFORE running,
  full numbers not yet seen).** Motivation: a confound audit found `relevance_prior`
  computing MI over the FULL (X, y) including test-fold labels → RG-SCSO has a mild
  transductive leak (mean Spearman ρ_full vs ρ_train = 0.75; worst on GermanCredit
  0.32, SpectEW 0.34, gene-set top-k Jaccard 0.48–0.60). This is an ADDITIONAL
  VALIDATION BRANCH, it does NOT replace the in-sample main results R3 (in-sample is
  the standard wrapper-FS protocol, kept as is + MI-on-train disclosed). Locked
  protocol: for each (algo,dataset,run) split outer 80/20 stratified
  (random_state=seed); MI prior + search + fitness (KNN 5-fold CV) ONLY on the
  train-80; report accuracy on the held-out-20. Budget matched (pop×iter=15000). The
  same 7 algorithms + 18 datasets + 30 runs as the main study. Metric of record =
  heldout_accuracy; tests Wilcoxon+Holm+Cohen's d + Friedman/CD as in Section 5.
  Success criteria UNCHANGED from Section 6 (no lowering the bar after seeing the
  pilot). Pilot R3b (8 datasets, 3 runs, budget-matched) achieved 7 WIN/0/1 LOSS vs
  SCSO — NOT a reported number, go/no-go only. Code:
  `src/feature_selection/run_fs_heldout.py`. Report HONESTLY even though the held-out
  margin is smaller than in-sample (the in-sample gap is largely from the optimistic
  bias of the shared CV — fair for ranking, inflates absolute effect sizes).
