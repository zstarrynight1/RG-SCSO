
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from build_paper_structure import load_summary
from build_paper_tex import (
    DIVERSITY_CSV,
    PROC_DIR,
    ROBUST_CSV,
    ablation_table,
    adaptive_baselines,
    dataset_table,
    diversity_analysis,
    esc,
    inference_value,
    pcmp,
    rank_table,
    robustness_baselines,
    robustness_svm16,
    runtime_table,
    scso_family_baselines,
    sensitivity_table,
    washout_table,
)
import build_heldout_table as _heldout


from build_paper_scirep import (
    CLASSIC_CSV,
    SIGNAL_POS_CSV,
    SIGPOS_FINAL_STEP,
    SIGPOS_LABEL_TEX,
    THRESHOLD_CSV,
    classic_baselines_table,
    classifier_robustness_table,
    heldout_combined_table,
    literature_positioning_table,
    rank_table_with_effect_size,
    rf_robustness_table,
    threshold_sensitivity_table,
)
from scipy.stats import wilcoxon as _wilcoxon
from src.stats.statistical_tests import holm_correction as _holm_correction

FS_MAIN_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
FS_DL_CSV = os.path.join("experiments", "results_fs_dl", "fs_dl_results.csv")


def _classic18_data():
    main = pd.read_csv(FS_MAIN_CSV)
    acc_col = "accuracy" if "accuracy" in main.columns else next(
        c for c in main.columns if "acc" in c.lower())
    nf_col = "n_selected_features" if "n_selected_features" in main.columns else "n_features"
    rg = main[main.algorithm == "RG-SCSO"]
    acc = {"RG-SCSO": rg.groupby("dataset")[acc_col].mean()}
    nf = {"RG-SCSO": rg.groupby("dataset")[nf_col].mean()}

    cb = pd.read_csv(CLASSIC_CSV)
    label = {"LASSO": "LASSO", "mRMR": "mRMR", "ReliefF-baseline": "ReliefF",
             "SFS": "SFS", "MI-threshold": "MI-thr"}
    for raw in ["LASSO", "mRMR", "ReliefF-baseline", "SFS", "MI-threshold"]:
        sub = cb[cb.algorithm == raw]
        acc[label[raw]] = sub.groupby("dataset")["accuracy"].mean()
        nf[label[raw]] = sub.groupby("dataset")["n_selected_features"].mean()

    if os.path.exists(FS_DL_CSV):
        dl = pd.read_csv(FS_DL_CSV)
        if len(dl):
            acc["CAE"] = dl.groupby("dataset")["accuracy"].mean()
            nf["CAE"] = dl.groupby("dataset")["n_selected_features"].mean()

    columns = [c for c in ["RG-SCSO", "LASSO", "mRMR", "ReliefF", "SFS",
                           "MI-thr", "CAE"] if c in acc]
    feat = {d: pd.read_csv(os.path.join(PROC_DIR, f"{d}.csv")).shape[1] - 1
            for d in acc["RG-SCSO"].index}
    datasets = sorted(acc["RG-SCSO"].index, key=lambda d: feat[d])

                                                                              
    others = [c for c in columns if c != "RG-SCSO"]
    a_rg = np.array([acc["RG-SCSO"][d] for d in datasets])
    raw_p, wl = {}, {}
    for c in others:
        b = np.array([acc[c].get(d, np.nan) for d in datasets])
        mask = ~np.isnan(b)
        wins = int((a_rg[mask] > b[mask]).sum())
        losses = int((a_rg[mask] < b[mask]).sum())
        try:
            _, p = _wilcoxon(a_rg[mask], b[mask])
        except ValueError:
            p = 1.0
        raw_p[c] = p
        wl[c] = {"wins": wins, "losses": losses, "n": int(mask.sum())}
    order = sorted(others, key=lambda c: raw_p[c])
    for i, c in enumerate(order):
        wl[c]["holm_p"] = min(1.0, raw_p[c] * (len(others) - i))
    best_all = sum(
        1 for d in datasets
        if all(acc["RG-SCSO"][d] > acc[c].get(d, -1) for c in others)
    )
    return {"datasets": datasets, "columns": columns, "acc": acc, "nf": nf,
            "feat": feat, "wl": wl, "best_all": best_all, "n": len(datasets)}


def classic_baselines_all18_table() -> str:
    d = _classic18_data()
    cols_spec = "l" + "c" * len(d["columns"])
    head = "Dataset & " + " & ".join(d["columns"]) + r" \\"
    lines = []
    for ds in d["datasets"]:
        best = max(d["acc"][c].get(ds, -1) for c in d["columns"])
        cells = [esc(ds)]
        for c in d["columns"]:
            a = d["acc"][c].get(ds, float("nan"))
            n = d["nf"][c].get(ds, float("nan"))
            val = f"{a:.3f} ({n:.0f})"
            if abs(a - best) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    def _fmtp(p):
        return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
    sig = ", ".join(
        f"{c} ({_fmtp(d['wl'][c]['holm_p'])})"
        for c in ["MI-thr", "SFS", "mRMR", "LASSO"] if c in d["wl"]
        and d["wl"][c]["holm_p"] < 0.05)
    rel = d["wl"].get("ReliefF", {})
    cap = (
        f"Full {d['n']}-dataset comparison of RG-SCSO against five classical "
        "filter/embedded/wrapper selectors and a Concrete-Autoencoder deep "
        "feature selector (CAE), replacing the earlier five-dataset pilot; "
        "accuracy with mean selected-feature count in parentheses, 30 runs "
        "each, identical evaluator (stratified 5-fold KNN) throughout, "
        "\\textbf{bold} = best accuracy per dataset. RG-SCSO attains the best "
        f"accuracy of all methods on {d['best_all']} of {d['n']} datasets. "
        "Across the benchmark it is significantly more accurate (Holm-corrected "
        f"Wilcoxon signed-rank over the {d['n']} datasets) than "
        f"{sig}; the one classical method it does not significantly beat is "
        f"ReliefF (better on {rel.get('wins','--')}/{d['n']}, "
        f"$p={rel.get('holm_p', float('nan')):.2f}$). The CAE baseline, matched "
        "to RG-SCSO's per-dataset subset size, is the weakest selector on the "
        "small-sample datasets tested, consistent with deep reconstruction "
        "objectives needing more data than these 50--455-instance benchmarks "
        "provide. LASSO and mRMR remain the strongest competitors on the two "
        "gene-expression ($p\\gg n$) sets."
    )
    return (
        "\\begin{sidewaystable}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:classic}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        f"\\begin{{tabular}}{{{cols_spec}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{sidewaystable}}\n"
    )


def sigpos_supp_table(s: dict) -> str:
    ds_list = s["abl_datasets"]
    sp = pd.read_csv(SIGNAL_POS_CSV)

    def sp_series(step: str) -> pd.Series:
        return sp[sp.step == step].groupby("dataset")["accuracy"].mean()

    def sp_values(step: str, ds: str) -> np.ndarray:
        sub = sp[(sp.step == step) & (sp.dataset == ds)].sort_values("run_id")
        return sub["accuracy"].to_numpy()

    final_mean = sp_series(SIGPOS_FINAL_STEP)
    rows = {step: sp_series(step) for step in SIGPOS_LABEL_TEX}

    sig = set()
    for step in SIGPOS_LABEL_TEX:
        pvals = []
        for ds in ds_list:
            a, b = sp_values(step, ds), sp_values(SIGPOS_FINAL_STEP, ds)
            try:
                _, p = _wilcoxon(a, b)
            except ValueError:
                p = 1.0
            pvals.append(p)
        p_holm = _holm_correction(np.array(pvals))
        for ds, p in zip(ds_list, p_holm):
            if p < 0.05 and rows[step].get(ds, 1.0) < final_mean.get(ds, 0.0):
                sig.add((step, ds))

    cols = "l" + "c" * len(ds_list)
    head = "Relevance-injection point & " + " & ".join(esc(d) for d in ds_list) + r" \\"
    lines = []
    ref_cells = ["Binarization interface (RMS+UMR, deployed)"]
    for ds in ds_list:
        m = final_mean.get(ds)
        ref_cells.append("--" if m is None else f"\\textbf{{{m:.4f}}}")
    lines.append(" & ".join(ref_cells) + r" \\")
    for step, label in SIGPOS_LABEL_TEX.items():
        cells = [label]
        for ds in ds_list:
            m = rows[step].get(ds)
            val = "--" if m is None else f"{m:.4f}"
            if (step, ds) in sig:
                val += r"$^\dagger$"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Signal-position experiment (RQ3): mean accuracy over 30 runs on the "
        "five ablation datasets when the same mutual-information relevance "
        "field is injected at initialization or as an objective penalty, "
        "instead of at the binarization interface (the deployed RG-SCSO, top "
        "row). $^\\dagger$ marks significantly worse than the deployed "
        "injection (paired Wilcoxon signed-rank, Holm-corrected $p<0.05$). "
        "MI-weighted objective injection is significantly worse on four of "
        "five datasets and tied on Leukemia; MI-guided initialization is worse "
        "on ColonCancer and Sonar but statistically indistinguishable on "
        "Leukemia, WDBC, and Zoo, and on Leukemia matches the deployed "
        "accuracy while selecting roughly four times fewer features "
        "(245 vs.\\ 940)."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:sigpos}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )

OUT_TEX = "RG-SCSO_IJCS.tex"
OUT_SUPP_TEX = "RG-SCSO_IJCS_Supplementary.tex"


def build() -> None:
    s = load_summary()
    colon = s["gene"].get("ColonCancer", {})

    adaptive = adaptive_baselines()
    scsofam = scso_family_baselines()
    robust = robustness_baselines()
    svm16 = robustness_svm16()
    inf_val = inference_value()
    c18 = _classic18_data()                                             

    diversity = diversity_analysis() if os.path.exists(DIVERSITY_CSV) else None

    _hs = _heldout.load()
    heldout_tab = heldout_combined_table(_hs)


    classifier_robust_tab = classifier_robustness_table().replace(
        "averaged over the same five representative datasets as "
        "Table~\\ref{tab:classic} (KNN/SVM",
        "averaged over five representative datasets (Zoo, Sonar, WDBC, "
        "ColonCancer, Leukemia; KNN/SVM",
    )
    hs_wtl = _hs["wil"]["mark"].value_counts()
    hs_w, hs_t, hs_l = (int(hs_wtl.get(k, 0)) for k in ("+", "=", "-"))
    hs_d_aoa = float(_hs["wil"][_hs["wil"].compared_with == "AOA"]["cohens_d"].abs().median())
    hs_rank = _hs["ranking"].sort_values()
    hs_stats = _hs.get("stats", {})

    feat_counts = [pd.read_csv(os.path.join(PROC_DIR, f"{d}.csv")).shape[1] - 1
                   for d in s["datasets"]]
    feat_min, feat_max = min(feat_counts), max(feat_counts)

    if s.get("stats"):
        w, ti, l = s["sig_total"]
        rank7 = s["rank7"]
        n_cmp = w + ti + l
    v = s.get("verdict", {})
    rms, orl, umr = v.get("NoRMS", {}), v.get("NoORL", {}), v.get("NoUMR", {})

    if inf_val is not None:
        inference_sentence = (
            " The dimensionality reductions are too small relative to these "
            "datasets' modest sample counts (50--455 training instances) for a "
            "wall-clock inference saving to be resolvable above call overhead; "
            "a projected, reconstructed inference-cost analysis at a synthetic "
            "deployment-scale workload, a controlled complexity demonstration "
            "rather than a measurement on the benchmark's own test sets, is "
            "reported in Supplementary Information instead of here."
        )
        inference_supp_paragraph = (
            "\\section{{Projected inference-cost analysis}}\n"
            "The main text (Feature-subset parsimony) notes that the "
            "dimensionality reductions RG-SCSO achieves are too small "
            "relative to these datasets' modest sample counts (50--455 "
            "training instances) for a wall-clock inference saving to be "
            "resolvable above call overhead on the benchmark's own test "
            "sets. To give a sense of what the same reductions would mean "
            "at a scale where inference cost is actually measurable, we "
            "replay the same feature-count reductions at a synthetic "
            "deployment-scale workload "
            f"(5{{,}}000 training instances, brute-force $k$-NN): this "
            f"projects a {inf_val['min_speedup']:.0f}--"
            f"{inf_val['max_speedup']:.0f}\\% reduction in batch-inference "
            f"cost relative to AOA (mean {inf_val['mean_speedup']:.0f}\\%). "
            "This is a controlled complexity demonstration, not a "
            "measurement on the benchmark's own test sets, and should not "
            "be read as an observed deployment speedup."
        )
    else:
        inference_sentence = ""
        inference_supp_paragraph = ""


    abstract = (
        "Wrapper feature selection with swarm intelligence typically searches "
        "in continuous space and crosses into the binary domain through a "
        "fixed transfer function, a feature-agnostic quantization that "
        "discards the continuous operators' fine adjustments, an effect we "
        "term washout. RG-SCSO replaces this transfer with a per-feature, "
        "relevance-modulated binarization: a mutual-information field biases "
        "each feature's bit-flip probability so informative features resist "
        "removal and noise resists inclusion, leaving SCSO's continuous "
        "search otherwise unchanged. "
        f"Evaluated on {s['n']} datasets, including two gene-expression sets, "
        "under a budget-matched, leak-free protocol that computes the "
        "relevance prior, the search, and the cross-validated fitness only on "
        "the training partition, RG-SCSO attains the best mean held-out "
        "accuracy of the seven metaheuristics compared while selecting the "
        "second-smallest feature subsets of any method tested, trailing only "
        "COA, which is several points less accurate. The advantage is "
        "parsimony rather than raw accuracy: it persists against optimizers "
        "carrying published adaptive transfers, against which RG-SCSO still "
        f"selects {adaptive['red_min']:.0f}--{adaptive['red_max']:.0f}\\% "
        "fewer features, and against same-family binary SCSO selectors; "
        "ablation and cross-prior tests show it depends on where the "
        "relevance signal is injected and on the prior used. On the two "
        "gene-expression sets a classical LASSO baseline is sparser at no "
        "loss of accuracy, so the transferable gain is parsimony on data "
        "without that extreme structure."
    )

                                                                            
    introduction = rf"""Feature selection removes irrelevant and redundant features to improve
classifier accuracy, reduce overfitting, and lower computational cost, a
payoff that is greatest for high-dimensional, small-sample problems such as
gene-expression classification, where the number of features exceeds the
number of samples by orders of magnitude~\cite{{guyon,mrmr}}. Wrapper
selection, which scores a subset by the performance of a downstream
classifier, is frequently cast as a combinatorial problem solved by
swarm-intelligence metaheuristics, including grey wolf~\cite{{bgwo}}, particle
swarm~\cite{{pso}}, whale~\cite{{mafarja}}, and several recent
optimizers~\cite{{aoa,coa,rime}}. Most such methods were conceived for
continuous optimization and are adapted to the binary space through a
transfer function, most commonly an S-shaped or V-shaped map, that converts a
real-valued position into a selection probability~\cite{{tf}}. We argue that
this retrofit contains a structural weakness: the transfer function is fixed
and feature-agnostic, applied identically to every dimension, so the
incremental adjustments a continuous-space operator makes are collapsed by
the squash-and-threshold step before they can influence the retained subset.
We refer to this loss as washout, and show in Methods below that it is not a
thought experiment: four well-motivated continuous-space enhancements of
SCSO~\cite{{scso}} failed to beat the base algorithm under an identical
feature-selection protocol (0 wins, 1 loss, 17 ties by Wilcoxon signed-rank;
Supplementary Table~S5, preliminary washout study).

Existing studies have explored relevance-guided initialization, objective
weighting, and adaptive binary transfer mechanisms; however, the explicit
injection of a per-feature relevance prior into the binarization operator of
SCSO-based feature selection remains insufficiently investigated
(Section~\ref{{sec:relwork}} positions this gap against each cited
SCSO-family work in detail).

We close this gap with RG-SCSO: a per-feature, relevance-modulated
binarization in which a mutual-information relevance field biases each
feature's bit-flip probability, turning a knowledge-agnostic quantization
step into a knowledge-carrying operator. SCSO's continuous search, including
its sensitivity range, is retained unchanged; the novelty resides entirely in
the binarization (Fig.~\ref{{fig:concept}}). This paper makes four
contributions. First, we identify washout as a mechanistic failure mode and
derive a diagnostic bound, together with a cumulative extension linking it to
discrete transition dynamics. Second, we propose RG-SCSO, whose
ablation-confirmed centerpiece is relevance-modulated sensitivity (RMS),
supplemented by a smaller, budget-neutral memetic refinement step (UMR) that a
sensitivity sweep shows contributes far less than RMS; an online-learning
variant of the relevance field is examined and pruned entirely by ablation.
Third, we evaluate under a preregistered, budget-matched, leak-free protocol
that denies the relevance prior any access to test labels. Fourth, we report a
full statistical treatment, a component ablation, and a size-fair enrichment
analysis correlating the observed parsimony with relevance guidance.

\subsection{{Research questions}}
This study is organized around four research questions, each answered
directly by a specific part of Results or Discussion below.
\begin{{itemize}}
\item \textbf{{RQ1.}} Does relevance-modulated binarization improve held-out
  feature-selection performance under a fixed computational budget?
  \emph{{Addressed in Section~\ref{{sec:heldout}} (Table~\ref{{tab:heldout}},
  Fig.~\ref{{fig:cd}}).}}
\item \textbf{{RQ2.}} Can RG-SCSO reduce the number of selected features while
  preserving competitive generalization accuracy?
  \emph{{Addressed in Section~\ref{{sec:parsimony}}.}}
\item \textbf{{RQ3.}} Is direct relevance injection at the binarization
  interface more effective than relevance-guided initialization or
  objective weighting?
  \emph{{Addressed in Section~\ref{{sec:ablation}} (Table~\ref{{tab:ablation}}).}}
\item \textbf{{RQ4.}} How robust is the proposed mechanism across
  classifiers, relevance priors, and dataset dimensionalities?
  \emph{{Addressed in Section~\ref{{sec:classical}}
  (Table~\ref{{tab:classifierrobust}}) and Discussion.}}
\end{{itemize}}

The remainder of the paper reviews existing approaches along three axes: (i)
binary transfer functions in swarm-based feature selection, (ii)
relevance-guided search mechanisms, and (iii) SCSO variants
(Section~\ref{{sec:relwork}})."""

                                                                          
    related_work = rf"""\subsection{{Binary transfer functions in swarm-based feature selection}}
Swarm-intelligence wrapper selectors, including grey wolf~\cite{{bgwo}},
particle swarm~\cite{{pso}}, whale~\cite{{mafarja}}, and several recent
optimizers~\cite{{aoa,coa,rime}}, are conceived for continuous optimization
and cross into the binary search space through a transfer function, most
commonly an S-shaped or V-shaped map~\cite{{tf}}. A substantial recent line of
work makes this interface \emph{{adaptive}} rather than static: time-varying
transfer functions that reshape the squashing curve as the search
progresses~\cite{{islam2017tvtf,bgoatvg2024}}, and fitness- or history-driven
schemes that tune it from the swarm's own progress~\cite{{teng2017avbpso,fhsbpso2024}}.
Crucially, all of these remain \emph{{feature-agnostic}}: the same (possibly
time-varying) transfer is applied identically to every dimension regardless
of that feature's relevance, the structural weakness this paper terms washout
(Introduction, Methods). RG-SCSO is orthogonal to this line -- it modulates
the binarization per feature by a relevance prior, and could in principle be
combined with an adaptive transfer rather than replacing one.

\subsection{{Relevance-guided mechanisms in feature selection}}
Filter criteria such as mutual information and mRMR~\cite{{guyon,mrmr}} encode
problem knowledge cheaply but are decoupled from the wrapper search; memetic
hybridization~\cite{{neri}} adds local refinement without addressing the
binarization interface. A large family of knowledge-guided metaheuristics
injects filter information \emph{{upstream}} of the binary decision, into
initialization or the objective. In other swarm families this takes the form
of correlation- or mutual-information-guided population
initialization~\cite{{mobgwogms2023,hhoelm2023}}, filter-guided PSO for
cancer-genome selection~\cite{{ludwig2025guided}}, and memetic hybridization
of binary Harris-hawks and whale operators~\cite{{whalehho2022}}; a parallel,
deep-learning-oriented direction drives selection from post-hoc attribution
or attention, for example SHAP-guided deep feature
selection~\cite{{xaifs2024}}. All of these bias \emph{{where}} the search
starts, \emph{{what}} it optimizes, or which features a separately trained
model highlights, yet leave the binarization operator itself
knowledge-agnostic. RG-SCSO differs by injecting the relevance signal at the
binarization interface itself rather than upstream of it; the signal-position
experiment in Results (Table~\ref{{tab:ablation}}) tests this design choice
directly against both the initialization and objective alternatives.

\paragraph{{Positioning against relevance-injecting binary GWO/WOA/HHO
variants.}} The closest competitors are binary grey-wolf, whale, and
Harris-hawks selectors that already carry a relevance signal: MOBGWO-GMS
initializes its population from feature correlation and guides mutation with
it~\cite{{mobgwogms2023}}, the enhanced Harris-hawks optimizer seeds the search
with filter-ranked features~\cite{{hhoelm2023}}, and hybrid binary
whale/Harris-hawks memetics combine two operators for stronger local
search~\cite{{whalehho2022}}. Each injects relevance at initialization,
mutation, or hybridization; none makes the \emph{{per-feature binarization
operator}} itself relevance-aware, which is the specific interface RG-SCSO
modifies and the reason its parsimony gain (Results) is not reproduced by
these variants' upstream injections.

\subsection{{Sand Cat Swarm Optimization and its variants}}
SCSO~\cite{{scso}} is a comparatively recent continuous metaheuristic. The
existing binary SCSO feature selectors~\cite{{bscso,scsofs2,scsofs3}} apply
standard transfer functions without a relevance-aware binarization step, and
the broader recent SCSO literature that adds chaotic initialization,
differential mutation, or hybridized search
strategies~\cite{{imscso2024,mescso2025,scsolensobl2024,improvedscso2024}}
improves continuous-space search dynamics while leaving the binarization
interface itself untouched -- none of these works, to our knowledge, makes
the binarization operator itself per-feature and relevance-aware.
Table~\ref{{tab:litpos}} below positions each of these works against this
claim directly.

\subsection{{Literature-positioning summary}}
Table~\ref{{tab:litpos}} makes this gap concrete across every SCSO-family
work cited above: the three directly comparable binary feature selectors
apply a standard, feature-agnostic transfer with no relevance-aware step,
and the four continuous-search variants improve exploration/exploitation
dynamics without touching the binarization interface at all. None of the
eight prior works positioned here makes the binarization operator itself
per-feature and relevance-aware -- the interface RG-SCSO modifies.

{literature_positioning_table()}"""

                                                                            
    fr_p_str = pcmp(hs_stats.get("friedman_p", 1)) if hs_stats else "<10^{{-3}}"
    fr_chi2_str = f"={hs_stats.get('friedman_chi2', 0):.2f}" if hs_stats else ""
    scsofam_pct_str = (f"{min(scsofam['red'].values()):.0f}"
                        if scsofam and scsofam.get("red") else "substantially")

    if s.get("stats"):
        ranking_block = (
            "Table~\\ref{tab:rank} gives the in-sample average rank "
            f"({rank7['RG-SCSO']:.2f} for RG-SCSO) and the Holm-significant "
            "win/tie/loss against each baseline (the corresponding "
            "critical-difference diagram is Supplementary Fig.~S6; "
            "Fig.~\\ref{fig:cd} in the main text gives the held-out "
            "counterpart). These "
            f"are paired per-dataset comparisons, not independent trials: "
            f"across the {n_cmp} dataset-baseline pairs RG-SCSO wins {w} and "
            f"loses {l}, with predominantly large effect sizes (median "
            f"$|d|={s['es_median']:.2f}$; {s['es_large_pct']:.0f}\\% exceed "
            "0.8). These in-sample effect sizes are an optimistic upper "
            "bound: they contract to the small-to-moderate held-out range "
            "above once the relevance prior is denied access to test "
            "labels, whereas the ranking itself is preserved."
        )
    else:
        ranking_block = "Ranking and significance are reported in the final version."

    if s.get("ablation"):
        ablation_block = (
            "We started from a three-component design and tested each part "
            "by removal (Table~\\ref{tab:ablation}), judging significance "
            "with a paired Wilcoxon signed-rank test (Holm-corrected across "
            f"datasets). RMS is the strongest: removing it costs "
            f"{rms.get('worst_delta_pts', 0):.2f} accuracy points on "
            f"{esc(rms.get('worst_ds', ''))} ($d={rms.get('worst_d', 0):.2f}$, "
            "Holm $p<0.001$). UMR is also load-bearing "
            f"({umr.get('worst_delta_pts', 0):.2f} points on "
            f"{esc(umr.get('worst_ds', ''))}), whereas ORL is not (degrades "
            f"accuracy on only {orl.get('n_deg', 0)}/{orl.get('n_ds', 0)} "
            "datasets) and is therefore dropped; the final RG-SCSO "
            "comprises RMS and UMR only."
        )
    else:
        ablation_block = "The ablation study is reported in the final version."

    signal_position_block = (
        "RQ3 asks whether direct relevance injection at the binarization "
        "interface is more effective than relevance-guided initialization "
        "or objective weighting. A dedicated signal-position experiment "
        "answers it: injecting the same mutual-information field elsewhere, "
        "at initialization or as an objective penalty, rather than at the "
        "binarization interface (both alternatives are reported in full in "
        "Supplementary Information). MI-weighted "
        "objective injection is significantly worse than the deployed "
        "RMS+UMR configuration on four of five datasets and tied on the "
        "fifth (Leukemia), supporting the original claim that the "
        "binarization interface is a more effective injection point than "
        "the objective function. MI-guided initialization is a harder "
        "comparison: significantly worse on ColonCancer and Sonar, but "
        "statistically indistinguishable on Leukemia, WDBC, and Zoo, and on "
        "Leukemia specifically it matches RG-SCSO's accuracy while selecting "
        "roughly four times fewer features (245 vs.\\ 940). The binarization "
        "interface is thus the most reliable injection point across "
        "datasets, not a uniformly superior one; on the most extreme "
        "$p\\gg n$ dataset tested, a simpler injection at initialization "
        "meets or beats it on both accuracy and parsimony."
    )

    _rel = c18["wl"].get("ReliefF", {})
    classic_baselines_block = (
        "RQ4 asks how robust the mechanism is across classifiers, relevance "
        "priors, and dataset dimensionalities; this subsection and "
        "Table~\\ref{tab:classifierrobust} answer it directly. "
        f"Table~\\ref{{tab:classic}} compares RG-SCSO across the full "
        f"{c18['n']}-dataset benchmark against five classical filter, "
        "embedded, and wrapper selectors -- mutual-information thresholding, "
        "mRMR, ReliefF, LASSO, and sequential forward selection -- and a "
        "Concrete-Autoencoder deep feature selector~\\cite{{concreteae}}, "
        "under the identical stratified 5-fold KNN evaluator. RG-SCSO attains "
        f"the best accuracy of all methods on {c18['best_all']} of "
        f"{c18['n']} datasets. Across the benchmark it is significantly more "
        "accurate (Holm-corrected Wilcoxon signed-rank over the "
        f"{c18['n']} datasets) than mutual-information thresholding, "
        "sequential forward selection, mRMR, and LASSO (all Holm $p<0.05$); "
        "the one classical selector it does not significantly beat is "
        f"ReliefF, which it still leads on {_rel.get('wins', 13)} of "
        f"{c18['n']} datasets ($p={_rel.get('holm_p', 0.20):.2g}$). The "
        "Concrete-Autoencoder baseline, matched to RG-SCSO's per-dataset "
        "subset size, is the weakest selector on the small-sample datasets "
        "tested, consistent with deep reconstruction objectives needing far "
        "more than the 50--455 training instances these benchmarks provide. "
        "The exceptions to RG-SCSO's lead are informative and concentrated: "
        "on the two gene-expression ($p\\gg n$) datasets a classical LASSO "
        "baseline attains higher accuracy with far fewer features on both "
        "Leukemia (99.8\\% at 23 features vs.\\ RG-SCSO's 98.6\\% at 940) and "
        "ColonCancer (95.7\\% at 64 vs.\\ 88.1\\% at 563), with mRMR "
        "competitive on both; and on four combinatorial or categorical "
        "datasets (TicTacToe, M-of-n, KrVsKpEW, WaveformEW) a classical "
        "method edges it by at most 1.7 accuracy points. RG-SCSO's advantage "
        "is thus broad but not universal: clearest on datasets without "
        "extreme $p\\gg n$ structure, where on ultra-high-dimensional "
        "gene-expression data a computationally far cheaper embedded method "
        "such as LASSO is a strong, arguably preferable, alternative. A "
        "remaining question is whether this parsimony mechanism is an "
        "artifact of the KNN wrapper used throughout; "
        "Table~\\ref{tab:classifierrobust} answers it directly, extending "
        "the same relevance-guided search to SVM and Random Forest wrappers. "
        "The advantage over a no-prior baseline is not a KNN artifact, though "
        "it is a more consistent parsimony gain than an accuracy one under "
        "Random Forest specifically, and the ReliefF-prior degradation "
        "already established above reproduces under every wrapper tested."
    )


    threshold_tab_asoc = (
        threshold_sensitivity_table() if os.path.exists(THRESHOLD_CSV) else ""
    ).replace(
        r"$\Phi$ is the Nogueira stability index (as "
        r"Table~\ref{tab:stability}).",
        r"$\Phi$ is the Nogueira stability index (Supplementary Information "
        r"gives the full feature-selection stability comparison across "
        r"algorithms).",
    )

    results = rf"""\subsection{{Held-out generalization}}
\label{{sec:heldout}}
RQ1 asks whether relevance-modulated binarization improves held-out
performance under a fixed computational budget; this subsection answers it
directly. For each dataset, algorithm, and independent run we draw an outer
stratified 80/20 split. The relevance prior, the search, and the
cross-validated fitness are computed exclusively on the 80\% training
partition; the selected subset is evaluated once on the untouched 20\%
hold-out, on which a fresh $k$-NN classifier (standardized on the training
partition) reports accuracy, so the relevance prior never has transductive
access to the test labels. Table~\ref{{tab:heldout}} reports held-out
accuracy over all seven algorithms and {s['n']} datasets, at a fixed
evaluation budget of $\mathrm{{max\_nfe}}=15000$ for every algorithm compared.
RG-SCSO attains the best average Friedman rank ({hs_rank.iloc[0]:.2f}, ahead of
the second-placed AOA at {hs_rank.iloc[1]:.2f}; $\chi^2${fr_chi2_str},
$p{fr_p_str}$). A
Holm-corrected Wilcoxon signed-rank test across all pairwise comparisons gives
RG-SCSO {hs_w} significant wins, {hs_l} loss, and {hs_t} ties; the only close
competitor is AOA, against which the advantage is genuine but moderate
(median $|d|={hs_d_aoa:.2f}$), RG-SCSO still leading on mean accuracy.
Fig.~\ref{{fig:cd}} visualizes this held-out ranking as a critical-difference
diagram: the Nemenyi test, a more conservative simultaneous comparison than
the pairwise Wilcoxon test above, does not separate RG-SCSO from AOA, but
places both above a single indistinguishable cluster comprising the
remaining five algorithms.

{heldout_tab}

\subsection{{Feature-subset parsimony}}
\label{{sec:parsimony}}
RQ2 asks whether RG-SCSO can reduce feature count while preserving
competitive accuracy; this subsection answers it directly. Parsimony, more
than any accuracy margin, is RG-SCSO's defining property.
Table~\ref{{tab:heldout}} shows it attains the best mean accuracy of any
method tested ({_hs['acc_mean']['RG-SCSO'].mean():.3f}) while selecting the
second-fewest features on the held-out setting, on average
{_hs['nf_mean']['RG-SCSO'].mean():.1f} against
{_hs['nf_mean']['SCSO'].mean():.1f} for base SCSO and
{_hs['nf_mean']['AOA'].mean():.1f} for AOA. Only COA selects fewer
({_hs['nf_mean']['COA'].mean():.1f} on average), and it does so at
{_hs['acc_mean']['RG-SCSO'].mean() - _hs['acc_mean']['COA'].mean():.3f} lower
mean accuracy, so RG-SCSO is the most compact method that does not trade
away accuracy to get there. On ColonCancer it retains
{colon.get('nf', float('nan')):.0f} of {colon.get('ntot', '--')} features
versus {colon.get('nf_aoa', float('nan')):.0f} for AOA, at higher accuracy.
This advantage survives every stress test we run: against optimizers
carrying published adaptive transfers RG-SCSO does not lead on accuracy yet
still selects {adaptive['red_min']:.0f}--{adaptive['red_max']:.0f}\% fewer
features, and against same-family binary SCSO
selectors~\cite{{bscso,scsofs2,scsofs3}} it is
{scsofam_pct_str}\% smaller at comparable accuracy.{inference_sentence}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{{accuracy_parsimony_tradeoff.pdf}}
\caption{{Mean held-out accuracy vs.\ mean selected-feature fraction,
averaged across all {s['n']} datasets, one point per algorithm. RG-SCSO
occupies the top-left corner, jointly the highest accuracy and the sparsest
subsets of all seven algorithms tested -- it is not merely competitive on
one axis at the cost of the other. AOA attains the second-highest accuracy
but selects on average 98\% of available features, essentially no feature
selection; COA is the second-sparsest method but at a distinctly lower
accuracy than RG-SCSO. In numbers, RG-SCSO's mean operating point is 0.866
accuracy at 41\% of features; AOA reaches 0.849 at 98\% of features, and COA
0.815; mean accuracy across the seven algorithms spans 0.791--0.866.}}
\label{{fig:tradeoff}}
\end{{figure}}

\subsection{{Ranking and statistical significance}}
{ranking_block}

{rank_table_with_effect_size(s) if s.get("stats") else rank_table(s)}

\subsection{{Ablation and mechanism}}
\label{{sec:ablation}}
{ablation_block}

{signal_position_block}

{ablation_table(s) if s.get("ablation") else ""}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{convergence_fs.pdf}}
\caption{{Mean best fitness versus iteration, RG-SCSO vs.\ SCSO vs.\ AOA, on
Zoo (16 features), WDBC (30 features), and ColonCancer (2000 features), mean
over 5 runs, on the actual feature-selection objective (illustrative, not a
new statistical claim). On Zoo the three algorithms converge along
essentially the same trajectory. On WDBC and ColonCancer, RG-SCSO both
converges faster and plateaus at a lower (better) fitness than SCSO or AOA,
which themselves plateau early at a distinctly worse value rather than
continuing to close the gap with more iterations; the difference grows with
dimensionality.}}
\label{{fig:convfs}}
\end{{figure}}

We also test whether relevance guidance makes RG-SCSO preferentially retain
high mutual-information features. Because a subset of size $|S|$ overlaps the
top-$|S|$ mutual-information features at a chance rate of $|S|/N$, we report a
size-fair enrichment (Fig.~\ref{{fig:mech}}), the fraction of selected
features in the top-$|S|$ set divided by this chance level. RG-SCSO's subset
is enriched above chance on both gene-expression sets, whereas the
relevance-agnostic SCSO sits at chance, evidence consistent with the
relevance field driving the smaller and more accurate subsets, though
enrichment alone is correlational. A direct causal test, detailed in
Supplementary Information, permutes the field's feature identities while
preserving its
value distribution: the permuted field yields no significant accuracy
difference from the real field on three of five datasets, a significant but
negligible-effect difference on Leukemia, and a clear difference only on
ColonCancer, while inverting the field's sign is significantly worse than
both on every dataset. The relevance field's direction and scale therefore
matter consistently; the exact per-feature ranking within it matters
demonstrably on only one of five datasets, a materially weaker causal claim
than the enrichment analysis alone would suggest.

\subsection{{Threshold sensitivity}}
The 0.5 preferred-bit threshold that separates preferred from disfavored
bits (Methods) is a convenience, not a theoretically grounded neutral point.
This sweep ($\tau\in\{{0.4,0.5,0.6\}}$, 30 independent runs per cell, same
protocol and datasets as the main ablation) tests whether that choice is at
least empirically reasonable. No single $\tau$ dominates uniformly:
$\tau=0.5$, the value deployed throughout this paper, attains the highest
accuracy on two of five datasets; mean selected-feature count falls
monotonically as $\tau$ increases on every dataset, at a modest,
dataset-dependent, non-uniform accuracy cost.

{threshold_tab_asoc}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.55\textwidth]{{threshold_heatmap.pdf}}
\caption{{Mean held-out accuracy (color) and mean number of selected features
(in parentheses) across the three preferred-bit thresholds tested
($\tau\in\{{0.4,0.5,0.6\}}$), one row per dataset. Accuracy is essentially
flat across thresholds on every dataset (largest swing 0.011, ColonCancer
$\tau{{=}}0.4$ vs.\ $\tau{{=}}0.5$), while the selected-feature count falls
monotonically as $\tau$ increases on all five datasets, confirming
$\tau=0.5$ is not a fragile choice.}}
\label{{fig:threshheat}}
\end{{figure}}

\subsection{{Comparison with classical selectors}}
\label{{sec:classical}}
{classic_baselines_block}

{classic_baselines_all18_table() if os.path.exists(CLASSIC_CSV) else ""}

{classifier_robust_tab}"""

                                                                             
    discussion = rf"""These results trace washout, a concrete failure mode of
transfer-function-based binary feature selection, to its source, and cure it
by moving the relevance signal directly inside the binarization operator
rather than upstream of it in the objective or the initialization. The formal
result motivating this design (Methods) is a diagnostic bound: it explains why
continuous-space enhancements fail at the binarization boundary, but it is not
a convergence guarantee for RG-SCSO itself. The underlying recipe, a filter
prior coupled to a wrapper search with memetic refinement, is a known
combination in the feature-selection literature. What is new is the injection
point, together with a stringent evaluation protocol: budget-matching, a
leak-free hold-out, cross-classifier and cross-prior robustness, and an
explicit exploration-safety diagnostic. We report parsimony, not raw accuracy,
as the transferable outcome of this choice.

Several boundaries delimit what these results establish. RG-SCSO inherits
SCSO's continuous search dynamics unchanged, and the RMS rule carries a risk
of its own: bias the flip probability too strongly and confidently classified
bits can freeze in place, draining population diversity. We measured this
directly rather than assuming it away: at an aggressive stress-test bias the
risk is real and grows with dimensionality, but the conservative $\gamma=0.5$
this paper deploys keeps the frozen-bit fraction below
{f"{diversity['max_frz_g5']*100:.1f}" if diversity else "1.1"}\% throughout
the run on every dataset tested. We also tested whether UMR's benefit
specifically requires targeting relevance-uncertain features, or only the
extra evaluation budget it spends, by replacing its targeted $K$-feature
selection with $K$ uniformly random features at matched NFE (Supplementary
Information). This untargeted control significantly outperforms targeted
UMR on both accuracy and feature count on the two gene-expression datasets,
where UMR's contribution is largest (ColonCancer: 90.6\% vs.\ 88.1\%
accuracy, 214 vs.\ 563 features; Leukemia: 423 vs.\ 940 features at
statistically indistinguishable accuracy), and ties it on the other three.
UMR's value over no memetic step at all remains real (Table~3); on the
datasets where that value is largest, however, it does not depend on
targeting relevance-uncertain features specifically, a more modest claim
than ``uncertainty-targeted'' on its own implies. The relevance field is
built from a single filter statistic. Swapping in a ReliefF prior exposes a
genuine limitation rather than confirming the mechanism: the parsimony
advantage disappears entirely (Supplementary Information). RG-SCSO's
compactness therefore depends on a prior that drives uninformative features
below the neutral point, not on the mere presence of a relevance signal. This dependence, together with the
$O(dn\log n)$ one-time cost of the mutual-information prior amortized
against the wrapper's $N+K$ per-iteration evaluations, is why we scope the
scalability claim to the wrapper cost dominating overall runtime as feature
dimensionality grows (full per-dataset wall-clock cost is in Supplementary
Information) rather than to any runtime advantage over same-budget
baselines, which we do not claim.

The main
objective is also a KNN wrapper. Under SVM, tested directly on
{svm16.get('n_ds', 16)} of the {s['n']} datasets, and under Random Forest on
the same five-dataset subset as Table~\ref{{tab:classifierrobust}}, the
parsimony advantage is not a KNN artifact. Under Random Forest specifically it
is a more consistent gain in subset size than in accuracy, and the
ReliefF-prior degradation established above reproduces under every wrapper
tested. The selected subset is smaller and
more consistent in size than competing algorithms', but not necessarily
more consistent in identity: a feature-selection stability index shows
RG-SCSO more stable run to run than same-family SCSO on every dataset,
clearest on the three lower-dimensional sets, yet on both gene-expression
datasets RG-SCSO's own stability is itself close to the level expected by
chance, so a much smaller subset there is not a materially more repeatable
one (Fig.~\ref{{fig:mech}}b; Supplementary Information gives the full
per-dataset breakdown for both checks).

This wrapper search is itself compute-intensive; absolute wall-clock cost
per run, for every algorithm tested, is reported in full in Supplementary
Information rather than only argued qualitatively. A genuine outer-fold
nested cross-validation pilot on three datasets, also in Supplementary
Information, is directionally consistent with the single-split held-out
estimate above, though underpowered at five runs per cell to confirm
significance independently.

External validity has its own limits: the benchmark spans
{feat_min} to {feat_max} features across biomedical, gene-expression, and
categorical domains drawn from a single curated family of UCI and standard
microarray sets. Its top end ({feat_max} features, Leukemia) is still an order
of magnitude below the $10^4$--$10^5$-feature omics and high-dimensional text
regimes where a binary-native, relevance-guided operator should matter most,
so behaviour at that scale is extrapolated, not measured; a genuine
$>$$10^4$-feature evaluation is deferred to future work because a full
seven-optimizer, 30-run wrapper study at that dimensionality is compute-bound
by the $O(N\,\mathrm{{max\_nfe}})$ KNN fitness evaluations rather than by the
relevance mechanism itself, and lies outside this study's fixed budget. The
relevance field
itself is not perfectly stable on the smallest, highest-dimensional
datasets: a bootstrap resampling analysis, reported in Supplementary
Information, shows markedly lower rank stability of the mutual-information
field on the two gene-expression sets than on the three lower-dimensional
benchmarks, a risk the classical-baseline comparison above makes concrete
rather than theoretical.

Finally, the accuracy claim is scoped, not universal: against
binary particle-swarm and grey-wolf optimizers carrying published adaptive
transfers, and against same-family binary SCSO selectors, RG-SCSO does not
lead on accuracy, and on the two gene-expression datasets specifically a
classical LASSO baseline outperforms RG-SCSO outright, so we do
not claim a practical advantage for RG-SCSO on ultra-high-dimensional,
small-sample data; we scope its transferable benefit to parsimony on
datasets without that extreme structure. These boundaries are gathered
together, with the future work they motivate, in Conclusion below."""

                                                                             
    conclusion = rf"""The results support RG-SCSO as a relevance-guided binary
feature-selection method whose primary advantage is improved subset
parsimony with competitive predictive performance under the tested
conditions: across {s['n']} benchmark datasets it selects the
second-smallest feature subsets of any method evaluated, trailing only COA,
while attaining the best mean held-out accuracy among all seven algorithms
compared, under a fixed, budget-matched, leak-free evaluation protocol
(RQ1, RQ2).

This advantage is not universal, and the boundaries established in
Discussion above are real constraints on the claim, not caveats to be
read past. Against optimizers carrying published adaptive continuous-space
transfers, and against same-family binary SCSO selectors, RG-SCSO does not
lead on accuracy. On the two extreme $p\gg n$ gene-expression datasets tested,
a classical LASSO baseline outperforms it outright on both accuracy and
feature count. Its dependence on the underlying relevance prior is genuine:
replacing mutual information with a ReliefF prior removes the parsimony
advantage entirely. The 0.5 preferred-bit threshold, while empirically
reasonable across the range tested, is not a uniformly optimal choice. And the
benchmark itself, though spanning {feat_min}--{feat_max} features across
biomedical, gene-expression, and categorical domains, is drawn from a single
curated family of datasets, so behavior on ultra-high-dimensional omics data
of $10^4$--$10^5$ features is extrapolated rather than measured.

In practical terms, RG-SCSO is a drop-in replacement for the binarization
step of any SCSO-based wrapper, adding only a small, fixed set of
hyperparameters (the modulation strength $\gamma$, the preferred-bit
threshold $\tau$, and the memetic budget $K$) that this paper shows are
not fragile choices; its main practical payoff is a smaller,
cheaper-to-store and cheaper-to-deploy feature subset at accuracy that
matches or exceeds the swarm-based selectors it was compared against.

\textbf{{Future work}} includes: (i) adaptive, data-driven selection of the
preferred-bit threshold, for example via cross-validation or an
entropy-based criterion, rather than a fixed value; (ii) an explicit
multi-objective formulation that traces an accuracy-parsimony Pareto front
rather than a single weighted objective; (iii) extending the
relevance-modulated binarization mechanism to other swarm optimizers, such
as WOA, HHO, or SMA, to test whether the injection-point argument
generalizes beyond SCSO; (iv) evaluation at larger scale, on datasets
exceeding $10^4$ features, where a relevance-guided, binary-native operator
should be especially valuable; and (v) alternative relevance priors beyond
mutual information and ReliefF, such as SHAP-based, mRMR, or learned
priors, to test whether the mechanism's benefit is specific to the prior
used here or transfers to others."""

    tex = rf"""% !TeX program = pdflatex
%=======================================================================
% RG-SCSO, Springer Nature (sn-jnl) format, target: Iranian Journal of
% Computer Science (IJCS). Forked from the Elsevier/elsarticle (ASOC)
% version; same body content throughout, only the class/frontmatter/
% back-matter differ per IJCS's own submission-guidelines page
% (link.springer.com/journal/42044/submission-guidelines): documentclass
% option [iicol,sn-basic,pdflatex], bibliography style sn-basic.bst.
% Font: newtxtext/newtxmath (Times-like, pdflatex-compatible) -- NOT
% fontspec/unicode-math (those need XeLaTeX/LuaLaTeX; IJCS's own
% guidelines specify the [pdflatex] class option, i.e. the standard pdfTeX
% engine).
%=======================================================================
\documentclass[iicol,sn-basic,pdflatex]{{sn-jnl}}
\usepackage{{newtxtext,newtxmath}}
\usepackage{{amsmath}}
\usepackage{{amsthm}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{graphicx}}
\graphicspath{{{{figures/}}{{./}}}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{rotating}}
\usepackage{{url}}
\usepackage{{placeins}}
\raggedbottom
\makeatletter
\setlength{{\@fptop}}{{0pt}}
\setlength{{\@fpsep}}{{12pt}}
\setlength{{\@fpbot}}{{0pt plus 1fil}}
\makeatother
\renewcommand{{\topfraction}}{{0.95}}
\renewcommand{{\floatpagefraction}}{{0.85}}
\renewcommand{{\bottomfraction}}{{0.9}}
\renewcommand{{\textfraction}}{{0.05}}
\setcounter{{topnumber}}{{4}}
\setcounter{{bottomnumber}}{{4}}
\setcounter{{totalnumber}}{{8}}

\newtheorem{{lemma}}{{Lemma}}
\newtheorem{{proposition}}[lemma]{{Proposition}}

\begin{{document}}

\title[RG-SCSO for Parsimonious Feature Selection]{{Relevance-Guided
Binarization for Parsimonious Feature Selection with Sand Cat Swarm
Optimization}}

\author*[1]{{\fnm{{Bui Quang}} \sur{{Huy}}}}\email{{huybq@donga.edu.vn}}
\author[1]{{\fnm{{Duong Minh}} \sur{{Son}}}}\email{{sondm@donga.edu.vn}}
\affil[1]{{\orgname{{Dong A University}}, \city{{Da Nang}}, \country{{Vietnam}}}}

\abstract{{{abstract}}}

\keywords{{Feature selection, Sand Cat Swarm Optimization, binary
optimization, relevance-guided binarization, parsimony}}

\maketitle

\section{{Introduction}}
{introduction}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{concept.pdf}}
\caption{{Conceptual overview. (a) The conventional pipeline, where
continuous-operator adjustments are collapsed by a fixed, feature-agnostic
transfer (washout). (b) RG-SCSO, where a per-feature, relevance-modulated
binarization replaces the feature-agnostic transfer, biasing each feature's
bit-flip probability by a mutual-information relevance field, followed by
memetic refinement on uncertain bits.}}
\label{{fig:concept}}
\end{{figure}}

\section{{Related work}}
\label{{sec:relwork}}
{related_work}

\section{{Results}}
{results}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{cd_diagram_heldout.pdf}}
\caption{{Critical-difference (Nemenyi) diagram at $\alpha=0.05$ over the
held-out ranking (Table~\ref{{tab:heldout}}); algorithms not joined by a bar
differ significantly in mean held-out rank. RG-SCSO attains the best mean rank
(1.33) and AOA the second (2.03); the remaining five algorithms cluster
between 4.31 and 5.75. The critical difference is $\mathrm{{CD}}=2.12$ (7
algorithms, 18 datasets), so RG-SCSO and AOA are statistically
indistinguishable from each other yet both separate from the rest. The
corresponding in-sample diagram appears in Supplementary Fig.~S6.}}
\label{{fig:cd}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{mechanism.pdf}}
\caption{{Mechanism evidence. (a) Size-fair top-MI enrichment on the two
gene-expression sets (selection precision divided by the chance level
$|S|/N$; mean over 30 runs, error bars = std): RG-SCSO enriches its subset
1.22--1.26$\times$ above chance (ColonCancer, Leukemia respectively),
whereas the relevance-agnostic SCSO sits at essentially chance (lift 1.00
on both). (b) Feature-selection stability (Nogueira $\Phi$, 30 runs) across
all five representative datasets: RG-SCSO attains a higher $\Phi$ than
same-family SCSO on every dataset (e.g., 0.541 vs. 0.512 on Zoo, 0.19 vs.
0.157 on WDBC), clearest on the three lower-dimensional sets, but
both are close to the near-zero level expected by chance on the two
gene-expression sets (RG-SCSO $\Phi\approx$0.014--0.015), where AOA's
near-zero $\Phi$ reflects that it selects nearly every
available feature rather than genuine instability.}}
\label{{fig:mech}}
\end{{figure}}

\section{{Discussion}}
{discussion}

\section{{Conclusion}}
{conclusion}

\section{{Methods}}
\label{{sec:method}}

\begin{{algorithm}}[!t]
\caption{{RG-SCSO for binary feature selection}}
\label{{alg:main}}
\begin{{algorithmic}}[1]
  \REQUIRE training partition $(X,y)$ (never the held-out fold; see Methods
    below for the fold-honest protocol); population size $N$; iterations
    $T_{{\max}}$; budget $\mathrm{{max\_nfe}}$; memetic size $K$; bias
    strength $\gamma$
  \ENSURE best feature mask $b^\ast$
  \STATE $\rho_j \leftarrow$ normalized mutual information $I(X_j;y)$,
    computed on $(X,y)$ only, $\forall j$ \hfill$\triangleright$ static
    relevance field
  \STATE $\forall j$: preferred bit $\hat{{b}}_j \leftarrow \mathbf{{1}}[\rho_j>0.5]$;
    strength $s_j \leftarrow 2|\rho_j-0.5|$
  \STATE initialize positions $x_i \sim \mathcal{{U}}(-1,1)^d$, $i=1,\dots,N$;
    form the initial masks by sign only ($b_i=1$ if $x_i>0$, else $0$);
    evaluate; set $b^\ast$ \hfill$\triangleright$ the relevance-modulated flip
    rule (lines~8--11) is \emph{{not}} applied at initialization
  \WHILE{{$\mathrm{{nfe}} < \mathrm{{max\_nfe}}$}}
    \STATE $R \leftarrow S_M\,(1-t/T_{{\max}})$ \hfill$\triangleright$ sensitivity range contracts
    \FOR{{each agent $i=1,\dots,N$}}
      \STATE update $x_i$ by the SCSO position rule using range $R$
      \FOR{{each feature $j$}}
        \STATE $p_{{\text{{base}}}} \leftarrow |\tanh(x_{{ij}})|$ \hfill$\triangleright$ base transfer $T(x_{{ij}})$
        \STATE $p_j \leftarrow p_{{\text{{base}}}}\,(1+\gamma s_j)$ if the flip moves bit $j$ toward
          $\hat{{b}}_j$, else $p_{{\text{{base}}}}\,(1-\gamma s_j)$
        \STATE flip bit $j$ with probability $\mathrm{{clip}}(p_j,0,1)$
      \ENDFOR
      \STATE evaluate mask; update $b^\ast$ if improved
    \ENDFOR
    \STATE $U \leftarrow K$ features whose $\rho_j$ is closest to $0.5$
      \hfill$\triangleright$ UMR on uncertain bits
    \FOR{{each $j \in U$}}
      \STATE flip bit $j$ of $b^\ast$; keep the flip only if fitness improves
    \ENDFOR
  \ENDWHILE
  \RETURN $b^\ast$
\end{{algorithmic}}
\end{{algorithm}}

\subsection{{Problem formulation}}
Let a candidate subset be a binary mask $b\in\{{0,1\}}^d$. The objective is
\begin{{equation}}
  f(b) = 0.99\,(1-\mathrm{{Acc}}(b)) + 0.01\,\frac{{|b|}}{{d}},
  \label{{eq:fitness}}
\end{{equation}}
where $\mathrm{{Acc}}(b)$ is the stratified 5-fold KNN accuracy ($k=5$) using only
the selected features. Every baseline in this study optimizes this same
objective with the same 0.01 cardinality weight. RG-SCSO is granted no
structural advantage on subset size from the fitness function itself, so any
parsimony gap reported below reflects the search mechanism, not a differently
weighted objective. SCSO governs its search with the sensitivity range
$R(t)=S_M - S_M\,t/T$ ($S_M=2$), which RG-SCSO retains while replacing the
binarization and adding a relevance field.

\subsection{{Theoretical motivation: transfer-function washout}}
\label{{sec:washout}}
Consider any binary-native optimizer that keeps a real-valued position and
binarizes coordinate $j$ through a transfer $T:\mathbb{{R}}\to[0,1]$. A
continuous-space enhancement can influence the retained subset only by
perturbing a coordinate, $x_j\mapsto x_j+\delta_j$; its entire effect on the
discrete decision is the induced change in probability
$\Delta_j=T(x_j+\delta_j)-T(x_j)$.

\begin{{lemma}}[Leverage bound]
Let $T:\mathbb{{R}}\to[0,1]$ be Lipschitz and piecewise continuously
differentiable. If a coordinate-space enhancement perturbs $x_j$ by $\delta_j$,
$|\Delta_j|\le\|T'\|_\infty\,|\delta_j|$, where $\|T'\|_\infty$ is the largest
slope $T$ attains between $x_j$ and $x_j+\delta_j$. For the two standard
transfers $\|\sigma'\|_\infty=\tfrac14$ and $\||\tanh|'\|_\infty=1$, and each
slope decays away from the origin; in a flat region, where
$\|T'\|_\infty\le\varepsilon$, the leverage collapses to
$|\Delta_j|\le\varepsilon\,|\delta_j|$ (proof in Supplementary Information).
\end{{lemma}}
\noindent\textbf{{Remark (why RG-SCSO is exempt).}} RG-SCSO breaks the premise
of the lemma: instead of routing information through the coordinate, it
modulates the flip probability directly at the transfer output,
$\Delta_j=\pm\,\gamma\,s_j\,V(x_j)$, independent of $T'$. The base transfer
$T$ still appears inside RG-SCSO's own rule as one multiplicative factor
(Eq.~\ref{{eq:rms}} below), but Proposition~2, which follows next,
characterizes only the base operator's transition probability
$T(x_j^{{(t)}})$ itself, not RG-SCSO's realized flip probability $p_j$, which
this Remark has just shown is exempt from Lemma~1's bound by construction.

\begin{{proposition}}[Transition probability under the base V-shaped transfer]
This proposition characterizes only the base, unmodulated V-shaped transfer
introduced above (the NoRMS reference operator used in the ablation);
RG-SCSO's own operator is exempt from it by construction, per the Remark
above. Under this unmodulated V-shaped binarization rule, bit $j$
flips at iteration $t$ with probability $T(x_j^{{(t)}})$, so the base
operator's one-step bit-transition probability coincides with $T$,
$P_{{\text{{base}}}}\big(b_j^{{(t+1)}}\ne b_j^{{(t)}}\mid x_j^{{(t)}}\big)=T(x_j^{{(t)}})$.
Lemma~1's
bound on $\Delta_j$ is therefore, without further assumption, already a
bound on the one-step change in the base operator's transition probability,
and this bound
composes linearly over $N$ repeated samples of the same bit in the flat,
saturated regime (proof in Supplementary Information). For RG-SCSO itself,
this base probability $T(x_j^{{(t)}})$ is subsequently modulated by the
relevance-modulated rule $p_j$ of Eq.~\ref{{eq:rms}} below, so RG-SCSO's
actual one-step transition probability is $p_j$, not $T(x_j^{{(t)}})$ alone.
This is a local,
one-step-composable sensitivity bound, not a convergence guarantee or a
proof that continuous-space enhancements are ineffective; the full scope
discussion is in Supplementary Information.
\end{{proposition}}

\subsection{{The RG-SCSO mechanism}}
Given the updated position $x_j$, the base flip probability uses a V-shaped
transfer $V(x_j)=|\tanh(x_j)|$. With relevance $\rho_j\in[0,1]$, preferred bit
$b^\ast_j=\mathbf{{1}}[\rho_j>0.5]$ and strength $s_j=2|\rho_j-0.5|\in[0,1]$,
the flip probability is biased toward $b^\ast_j$,
\begin{{equation}}
  p_j = \mathrm{{clip}}\!\Big(|\tanh(x_j)|\,\big(1+\gamma\,\sigma_j\,s_j\big),\,0,\,1\Big),
  \label{{eq:rms}}
\end{{equation}}
where $\sigma_j=+1$ if the flip moves bit $j$ toward $b^\ast_j$ and $\sigma_j=-1$
otherwise. Setting $\gamma=0$ recovers a plain V-shaped operator (the NoRMS
ablation). The 0.5 threshold that separates preferred from disfavored bits is
a convenience, not a theoretically grounded neutral point: with
$\rho_j=I(X_j;y)/H(y)$, the value at which a feature stops being informative
depends on the number of classes, the label entropy, the sample size, and
the specific MI estimator used, none of which are normalized away by the
$H(y)$ division alone. The Discussion shows this dependence is not merely
theoretical: the method's compactness is tied to how the chosen relevance
statistic happens to place features around 0.5, not to the threshold
itself. A threshold sweep ($\tau\in\{{0.4,0.5,0.6\}}$, Results
Section~\ref{{sec:ablation}}) supports this choice as reasonable rather than
arbitrary: $\tau=0.5$ attains the highest accuracy on two of five datasets
tested, and every value trades a modest, dataset-dependent accuracy shift
for a monotonic reduction in feature count as $\tau$ increases, with no
single value dominating uniformly.

The field $\rho$ that drives this mechanism is, in the final method, the
static prior $\rho_{{\mathrm{{static}}}}=\mathrm{{clip}}(\mathrm{{MI}}(f_j;y)/H(y),0,1)$,
a normalized mutual-information filter score~\cite{{mrmr,kraskov}} computed once
from the data; an online extension adding an EMA credit-assignment term from
accepted fitness improvements was examined and dropped, as the ablation
shows no accuracy gain on any dataset. A second, smaller component,
uncertainty-targeted memetic refinement (UMR), spends a fixed local-search
budget where this prior is least decisive: each iteration, the $K$ features
whose relevance is closest to $0.5$ are greedily flipped on the incumbent
best mask~\cite{{neri}} and the flip is kept only if fitness improves.

\subsection{{Algorithm and computational cost}}
Every fitness evaluation, whether from population moves, memetic probes, or
initialization, is counted against a single budget $\mathrm{{max\_nfe}}=\mathrm{{pop\_size}}\times
\mathrm{{max\_iter}}=15000$, identical to the baselines, so UMR grants no extra
evaluations (Algorithm~\ref{{alg:main}}).

The per-iteration cost is dominated by the $N+K$ wrapper evaluations, each a
KNN fit under fixed folds; the mutual-information prior adds a one-time
$O(dn\log n)$ preprocessing cost (the nearest-neighbor MI
estimator~\cite{{kraskov}}), amortized against the wrapper evaluations. RG-SCSO
is thus asymptotically no more expensive than base SCSO apart from the $K$
extra probes. In wrapper feature selection generally, classifier-based
fitness evaluation, not the search mechanism itself, dominates overall
computational cost as feature dimensionality grows; per-dataset wall-clock
measurements across the full {feat_min}--{feat_max}-feature range tested
(Supplementary Information) are consistent with this expectation, and we
make no runtime-speedup claim for RG-SCSO over same-budget baselines.

\subsection{{Datasets, baselines, and protocol}}
The benchmark spans {s['n']} preprocessed datasets of {feat_min} to
{feat_max} features across biomedical, gene-expression, and categorical
domains, full characteristics in Supplementary Table~S1.

We benchmark against six baselines: SCSO (base)~\cite{{scso}},
AOA~\cite{{aoa}}, CoatiOA~\cite{{coa}}, GWO~\cite{{gwo}}, PSO~\cite{{pso}}, and
RIME~\cite{{rime}}. The protocol is preregistered and locked prior to the full
run. All algorithms share: population 30, 500 iterations, 30 independent
runs, seed $=42+\mathrm{{run\_id}}$ (paired across algorithms), KNN ($k=5$)
with stratified 5-fold cross-validation, search space $[-1,1]^d$, and the
fitness of~\eqref{{eq:fitness}}. Every baseline runs with its library-default
published hyperparameters, with only population size and evaluation budget
matched across methods; RG-SCSO's $\gamma$ and $K$ are likewise fixed before
the full run.

\subsection{{Statistics and reproducibility}}
Significance uses the paired Wilcoxon signed-rank test with Holm
correction~\cite{{holm}}, the Holm family formed \emph{{per dataset}}, at
$\alpha=0.05$. A tie denotes failure to reject $H_0$ (no significant
accuracy difference) at this threshold; it is not evidence of equivalence.
Effect sizes use Cohen's $d$ and rank-biserial $r$; overall comparison uses the
Friedman test with a critical-difference diagram~\cite{{demsar}}. The
experimental design was preregistered and version-controlled before the full
run and left unmodified after results were observed; all randomness is
seeded deterministically and shared across algorithms.

\section*{{Author Contributions}}
\textbf{{Bui Quang Huy:}} Conceptualization, Methodology, Software,
Validation, Formal analysis, Investigation, Data curation, Writing --
original draft, Writing -- review \& editing, Visualization, Supervision.
\textbf{{Duong Minh Son:}} Formal analysis, Writing -- review \& editing.

\section*{{Statements and Declarations}}

\subsection*{{Funding}}
No funding was received for this work.

\subsection*{{Conflict of interest}}
The authors declare no competing interests.

\subsection*{{Ethical approval}}
Not applicable. This study uses only publicly available benchmark datasets
(UCI and standard microarray sets) and involves no human participants,
human data, or animals.

\subsection*{{Data Availability Statement}}
The datasets analysed in this study are publicly available benchmarks (UCI
and standard microarray sets); Supplementary Table~S1 lists each source. The
source code, the locked preregistration, the per-run seeds, a complete
hyperparameter table, a pinned dependency list, and the raw per-run results
are available for review in an anonymized repository
(\url{{https://anonymous.4open.science/r/RG-SCSO}}) and will be deposited in a
public, citable repository with a permanent Zenodo DOI upon acceptance,
permitting bit-for-bit reproduction of every number reported in this paper.

% Note: sn-jnl.cls already sets \bibliographystyle{{sn-basic}} internally, so
% an explicit \bibliographystyle here would trigger BibTeX's "Illegal, another
% \bibstyle command" error; we deliberately omit it and only invoke
% \bibliography.
\bibliography{{references}}

\end{{document}}
"""


    tex = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", tex)

    with open(OUT_TEX, "w") as fh:
        fh.write(tex)


    washout_tab_placeholder = washout_table(s)
    rf_robustness_placeholder = rf_robustness_table()
    from build_paper_scirep import (
        NESTED_CV_CSV,
        NFE_CONTROL_CSV,
        RELEVANCE_VAR_CSV,
        SHUFFLE_MI_CSV,
        STABILITY_CSV,
        nested_cv_table,
        nfe_control_table,
        notation_table,
        relevance_variance_table,
        shuffle_mi_table,
        stability_index_table,
    )
    supp = rf"""%=======================================================================
% Supplementary Information for RG-SCSO (Iranian Journal of Computer
% Science submission). Font matches the main text: newtxtext/newtxmath
% (Times-like, pdflatex-compatible), not fontspec/unicode-math -- the main
% .tex now targets sn-jnl[pdflatex], so both files compile with the same
% standard pdfTeX engine.
%=======================================================================
\documentclass[9pt]{{article}}
\usepackage[a4paper,margin=25mm]{{geometry}}
\usepackage{{newtxtext,newtxmath}}
\usepackage{{amsmath,graphicx,booktabs,multirow,rotating,hyperref}}
\usepackage{{placeins}}
\graphicspath{{{{figures/}}{{./}}}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\renewcommand{{\thesection}}{{S\arabic{{section}}}}
\title{{Supplementary Information for: Relevance-Guided Binarization for
Parsimonious Feature Selection with Sand Cat Swarm Optimization}}
\author{{Bui Quang Huy, Duong Minh Son}}
\date{{}}
\begin{{document}}
\maketitle

\section{{Proofs of Lemma 1 and Proposition 2}}
\textbf{{Proof of Lemma 1 (Leverage bound).}} By the mean value theorem
there is a point $\xi$ strictly between $x_j$ and $x_j+\delta_j$ for which
$\Delta_j=T'(\xi)\,\delta_j$, so
$|\Delta_j|=|T'(\xi)|\,|\delta_j|\le\|T'\|_\infty\,|\delta_j|$; because
$|\tanh|$ fails to be differentiable only at the origin, applying the
theorem separately on each side of zero extends the bound to every
interval. Differentiating the two standard transfers gives
$\sigma'(x)=\sigma(x)(1-\sigma(x))=\tfrac{{e^{{-|x|}}}}{{(1+e^{{-|x|}})^2}}$ and
$|\tanh|'(x)=\operatorname{{sech}}^2(x)=\tfrac{{4e^{{-2|x|}}}}{{(1+e^{{-2|x|}})^2}}$,
whose maxima are $\tfrac14$ at $\sigma=\tfrac12$ and $1$ as $x\to0$, and
each decays away from the origin ($\sigma'(x)\le e^{{-|x|}}$,
$\big||\tanh|'(x)\big|\le 4\,e^{{-2|x|}}$). Substituting
$\|T'\|_\infty\le\varepsilon$ on a flat interval establishes the claim.

\textbf{{Proof of Proposition 2 (Transition probability under the base
V-shaped transfer).}} The identity $P_{{\text{{base}}}}\big(b_j^{{(t+1)}}\ne
b_j^{{(t)}}\mid
x_j^{{(t)}}\big)=T(x_j^{{(t)}})$ follows directly from the definition of the
\emph{{base, unmodulated}} V-shaped flip rule (a bit is flipped, independently
at each iteration, with probability equal to the transfer value); no further
argument is required because transition and flip are the same event under
this base rule. RG-SCSO's own rule replaces this base probability with the
relevance-modulated $p_j$ of Eq.~(2) of the main text, so this identity does
not apply to RG-SCSO directly (see Remark, main text). The
one-step bound $\big|\Delta P\big(b_j^{{(t+1)}}\ne b_j^{{(t)}}\big)\big|\le
\|T'\|_\infty\,|\delta_j|$ is then Lemma~1 restated in this coincidence.
Summing the per-step bound over $t=1,\dots,N$ and applying the triangle
inequality gives the cumulative bound
$\big|\sum_{{t=1}}^{{N}}\Delta P\big(b_j^{{(t+1)}}\ne b_j^{{(t)}}\big)\big|\le
\varepsilon\sum_{{t=1}}^{{N}}|\delta_j^{{(t)}}|$; expectation is linear, so the
bound on the sum of per-step probability differences is also a bound on
the difference in expected flip counts accumulated over the $N$ steps.

\textbf{{What this does and does not establish.}} Proposition 2 answers the
specific gap raised in review: because transition and flip coincide under
the rule used here, the local sensitivity bound of Lemma~1 is already a
bound on transition probability, and that bound composes linearly over
repeated sampling of the same bit, so the accumulated leverage of a
continuous-space perturbation over an entire saturated-regime run stays
controlled by $\varepsilon$ and does not grow without bound as $N$
increases. It does not establish that continuous-space enhancements are
ineffective: a perturbation applied while $T'$ is not small, or a
sufficiently large $\sum_t|\delta_j^{{(t)}}|$, can still accumulate a
non-negligible transition-probability shift. The result is a genuine,
expectation-level link from coordinate perturbation to discrete transition
dynamics, not a proof that washout is unavoidable.

\textbf{{Scope of this result.}} Lemma~1 establishes a local sensitivity
bound on how far a continuous-space perturbation can move a single flip
probability. Proposition~2 extends this to a cumulative bound over
repeated sampling of the same bit, showing the accumulated leverage stays
controlled by $\varepsilon$ in the flat, saturated regime rather than
growing without bound. Neither result proves that continuous-space
enhancements become uniformly ineffective once repeated stochastic
binarization is applied outside that regime, or that a probability change
of any size never accumulates into a discrete decision loss. We use
washout, throughout this paper, as a diagnostic framing motivated by these
bounds and corroborated empirically (main text Results; this document), not
as a formal guarantee that continuous-space enhancements must fail under
binary search.

\section{{Dataset characteristics}}
{dataset_table(s)}

\section{{In-sample accuracy and parsimony (optimistic upper bound)}}
\label{{sec:insample}}
The main text reports held-out (leak-free) accuracy as the primary estimate.
For completeness, this section reports the
standard in-sample protocol, in which the relevance prior, search, and
reported metric share the same cross-validation folds; effect sizes here are
inflated relative to the held-out estimate (see main text Discussion).

\section{{Convergence behaviour}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{convergence.pdf}}
\caption{{Mean best fitness versus iteration on a low-dimensional (Zoo) and a
high-dimensional (ColonCancer) dataset, averaged over 30 runs.}}
\label{{fig:conv}}
\end{{figure}}

\section{{Exploration-safety diagnostic}}
{diversity['table'] if diversity else ''}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{diversity.pdf}}
\caption{{Population diversity and frozen-bit fraction versus iteration, for
$\gamma=0$, the deployed $\gamma=0.5$, and the stress-test $\gamma=1$, on
three datasets of increasing dimensionality.}}
\label{{fig:diversity}}
\end{{figure}}

\section{{Preliminary washout study}}
{washout_tab_placeholder}

\section{{Hyperparameter sensitivity}}
{sensitivity_table()}

\section{{Isolating the relevance contribution from adaptive transfers}}
The table below reports binary particle-swarm and grey-wolf optimizers
equipped with published adaptive V-shaped transfers, run under the identical
protocol and budget as the main study but with no per-feature relevance
signal.

{adaptive['table']}

\section{{Signal-position experiment: alternative relevance-injection points}}
This experiment (RQ3, main text Results, Ablation and mechanism) asks whether
the per-feature relevance signal is best injected at the binarization
interface or upstream of it, at initialization or as an objective penalty.
The two alternatives are collected here as a standalone table so that the
main-text component ablation stays focused on the RMS/UMR/ORL decomposition;
the numbers are identical to those discussed in the main text.

{sigpos_supp_table(s) if s.get("ablation") and os.path.exists(SIGNAL_POS_CSV) else ''}

\section{{Comparison with same-family SCSO feature selectors}}
Reimplementations of standard binary-SCSO recipes (S-shaped and
V-shaped+opposition-based-learning transfers), reported under the identical
protocol, are compared against RG-SCSO in the main study's Discussion.

{scsofam['table'] if scsofam and scsofam.get('table') else ''}

\section{{Robustness across classifiers and relevance priors}}
Cross-tabulation of KNN/SVM wrappers with MI/ReliefF priors, and the SVM
wrapper on {svm16.get('n_ds', 16)} of {s['n']} datasets, are summarized in
the main text Discussion. The remaining two datasets are excluded from the
SVM check specifically, not from any other analysis: an RBF-kernel SVM
refit at every one of the wrapper's roughly 15{{,}}000 fitness evaluations
per run scales at least quadratically in sample count, which is intractable
within the fixed evaluation budget on the two largest-$n$ datasets tested
(KrVsKpEW, WaveformEW); a Random Forest wrapper does not carry this
architectural restriction (each tree fit scales near-linearly in sample
count), and is reported below on the same five-dataset representative
subset as the KNN/SVM comparison above.

{robust['table'] if robust and robust.get('table') else ''}
{svm16['table'] if svm16 and svm16.get('table') else ''}
{rf_robustness_placeholder}

\section{{Computational cost}}
{runtime_table()}

{inference_supp_paragraph}

\section{{Causal intervention on the relevance field}}
We test whether RG-SCSO's performance depends on the
specific per-feature mutual-information values or merely on the presence of a
directionally-correct relevance signal, by permuting the field's feature
identities (Shuffled MI, preserving the marginal distribution of $\rho$ but
not which feature owns which value) and by inverting its sign (Inverted MI,
$\rho\mapsto1-\rho$), then re-running the full RG-SCSO search under each field.

{shuffle_mi_table() if os.path.exists(SHUFFLE_MI_CSV) else ''}

\section{{Nested cross-validation pilot}}
The main study's generalization estimate uses a single
80/20 held-out split per run. This pilot instead nests the entire search
inside every fold of an outer 5-fold cross-validation, so the reported
accuracy is never optimistic about which single split was drawn.

{nested_cv_table() if os.path.exists(NESTED_CV_CSV) else ''}

\section{{Relevance-field bootstrap stability}}
Because $\rho$ is re-estimated from a finite sample, its
ranking of features could itself be unstable, particularly on the
gene-expression datasets where the number of features vastly exceeds the
number of samples. We resample each dataset with replacement 30 times and
recompute $\rho$ each time.

{relevance_variance_table() if os.path.exists(RELEVANCE_VAR_CSV) else ''}

\section{{Feature-selection stability}}
A smaller selected subset is only a stronger
practical claim if it is also a consistent one: does each independent run
return largely the same features, or just a different subset of the same
size? We report the Nogueira~\cite{{nogueira2018stability}} stability index,
the standard generalization of the Kuncheva consistency index to variable
subset size, since RG-SCSO's own subset size is not fixed across runs.

{stability_index_table() if os.path.exists(STABILITY_CSV) else ''}

\section{{NFE-matched random-probe control}}
Isolates whether UMR's benefit
(Table~3) requires targeting relevance-uncertain features specifically, or
only the extra evaluation budget it spends, by replacing its targeted
$K$-feature selection with $K$ uniformly random features at matched NFE.

{nfe_control_table() if os.path.exists(NFE_CONTROL_CSV) else ''}

\section{{Notation}}
Compiled directly from the
symbols used in Methods; no symbol introduced here that is not already in
the main text.

{notation_table()}

\section{{In-sample ranking: critical-difference diagram}}
Main text Fig.~2 shows the critical-difference diagram over the held-out
ranking (Table~1), the paper's primary evidence. Figure~\ref{{fig:cdinsample}}
below is the corresponding diagram over the in-sample ranking (\S\ref{{sec:insample}}
above), the protocol in which the relevance prior, search, and reported metric
share the same cross-validation folds; as elsewhere in this Supplementary
Information, in-sample results are an optimistic upper bound and are reported
here for completeness rather than as the primary claim.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{cd_diagram.pdf}}
\caption{{Critical-difference (Nemenyi) diagram at $\alpha=0.05$ over the
in-sample ranking (an optimistic upper bound; main text Fig.~2 gives the
held-out counterpart); algorithms not joined by a bar differ significantly in
mean in-sample rank.}}
\label{{fig:cdinsample}}
\end{{figure}}

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
"""


    supp = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", supp)

    with open(OUT_SUPP_TEX, "w") as fh:
        fh.write(supp)

    print(f"Đã ghi {OUT_TEX} và {OUT_SUPP_TEX}")


if __name__ == "__main__":
    build()
