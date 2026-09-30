# RG-SCSO: Relevance-Guided Binarization for Parsimonious Feature Selection with Sand Cat Swarm Optimization

Code, configuration, and results supporting the manuscript "Relevance-Guided
Binarization for Parsimonious Feature Selection with Sand Cat Swarm
Optimization," submitted to *Applied Soft Computing*.

RG-SCSO injects a per-feature mutual-information relevance prior directly at
the continuous-to-binary interface of Sand Cat Swarm Optimization (SCSO),
rather than at initialization or via an objective-weighting term. The main
manuscript (`RG-SCSO_ASOC.docx`) reports held-out evaluation on 18 benchmark
datasets against six baselines (SCSO, AOA, COA, RIME, PSO, GWO).

## Requirements

- Python 3.10 (developed and tested on 3.10.14)
- Dependencies pinned in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository structure

```
config.py               Centralized experiment parameters (population size,
                         NFE budget, seeds, fitness weights) -- every script
                         imports from here rather than hardcoding values.
src/
  algorithms/            RG-SCSO, SCSO, and baseline optimizer implementations
  feature_selection/      Fitness function, transfer/binarization, run harnesses
  stats/                  Wilcoxon/Holm/Friedman statistical-test helpers
data/
  raw/                    Original benchmark datasets (UCI + microarray sets)
  processed/               Preprocessed CSVs consumed by the experiment harnesses
experiments/              Per-experiment output directories (raw per-run CSVs):
                         results_fs/ (main benchmark), results_fs_heldout/
                         (leak-free held-out protocol, the paper's primary
                         evidence), results_threshold/, results_nfe_control/,
                         results_convergence/, results_fs_shuffle_mi/,
                         results_fs_nested_cv/, results_stability/, and others
tests/                   Smoke tests (pytest)
make_figures.py          Regenerates every figure in figures/ from the
                         experiment CSVs (no numbers are hand-typed into figures)
build_paper_asoc.py      Generates the LaTeX manuscript + Supplementary
                         Information from the experiment CSVs
build_paper_asoc_docx.py Generates the Word manuscript (the version actually
                         submitted) from the same experiment CSVs
```

## Reproducing the results

All randomness is seeded deterministically: `seed = RANDOM_SEED_BASE + run_id`
(`RANDOM_SEED_BASE = 42` in `config.py`), shared identically across every
algorithm compared. Population size, iteration budget, and the resulting
fitness-evaluation budget (`max_nfe = population_size * max_iteration`,
locked at 15000) are likewise centralized in `config.py` and identical for
every method.

```bash
# Run the main 18-dataset x 7-algorithm x 30-run benchmark
python -m src.feature_selection.run_fs

# Run the leak-free held-out protocol (the paper's primary evidence -- the
# relevance prior, search, and fitness are computed only on the 80% outer
# training split; the held-out 20% is touched exactly once, for evaluation)
python -m src.feature_selection.run_fs_heldout

# Regenerate every figure from the experiment CSVs
python make_figures.py

# Regenerate the IJCS manuscript (LaTeX + Supplementary + Word) from the
# experiment CSVs. On macOS 26/27, scipy's prebuilt Fortran extensions fail
# to load; run the generators through the non-destructive wrapper, which
# stubs only those unused extensions (Wilcoxon/Friedman results are unaffected):
python build_run_macos27.py build_paper_ijcs.py        # -> .tex + Supplementary
python build_run_macos27.py build_paper_ijcs_docx.py   # -> generator .docx
# then compile the PDF (tectonic or Overleaf):
tectonic RG-SCSO_IJCS.tex && tectonic RG-SCSO_IJCS_Supplementary.tex
```

> **Note on the submission `.docx`.** `RG-SCSO_IJCS.docx` as submitted has been
> **hand-finalized in Word** — its 183 equations were converted to native Word
> equation objects, and margins/justification/hyphenation were set for the
> Springer template. `build_paper_ijcs_docx.py` therefore reproduces the *text
> and tables* but **not** the hand-edited equation rendering; the LaTeX/PDF
> (`build_paper_ijcs.py` → `RG-SCSO_IJCS.pdf`) is the fully reproducible source
> of record. Treat `RG-SCSO_IJCS.docx` as a frozen manual artifact.

Each harness supports a `--smoke` flag for a fast wiring check (1-2 runs,
not a result) before committing to a full run.

## Key methodological notes

- **MI relevance prior is leak-free.** For each outer 80/20 split, the
  mutual-information relevance field is computed exclusively from the 80%
  training partition and is never recomputed using, or given access to,
  the held-out 20%. See `src/feature_selection/run_fs_heldout.py`.
- **Fitness function** is shared identically across every algorithm
  compared: `f(b) = 0.99*(1 - Acc(b)) + 0.01*(|b|/d)`, stratified 5-fold
  KNN accuracy, `FITNESS_ALPHA`/`FITNESS_BETA` in `config.py`.
- **Statistics**: paired Wilcoxon signed-rank tests with Holm correction
  (per-dataset family), Friedman ranking with a Nemenyi critical-difference
  diagram, Cohen's *d* and rank-biserial *r* effect sizes
  (`src/stats/statistical_tests.py`).

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
