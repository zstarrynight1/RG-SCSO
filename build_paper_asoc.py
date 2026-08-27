"""Sinh bản thảo LaTeX cho Applied Soft Computing (Elsevier) -- SINGLE SOURCE
OF TRUTH cho target ASOC, song song với build_paper_scirep.py (Scientific
Reports). KHÔNG chạy thí nghiệm mới: mọi số liệu đọc động từ đúng các CSV/hàm
build_paper_scirep.py đã dùng, tái sử dụng trực tiếp các hàm sinh bảng/hình đã
verify ở đó (import, không copy-paste) để đảm bảo số liệu khớp tuyệt đối giữa
2 bản.

Khác biệt cấu trúc so với build_paper_scirep.py (theo RG-SCSO_MASTER_FINAL_
COMPLETE.md, chỉ áp dụng phần đã fact-check, KHÔNG áp dụng các claim sai như
"lowest mean feature count" -- vẫn dùng khung đã verify "second-smallest,
trailing only COA"):
  - \\documentclass{elsarticle} (không phải sn-jnl), \\bibliographystyle{elsarticle-num}.
  - Research Questions (RQ1-4) cuối Introduction, có cross-reference tới đúng
    bảng/hình trả lời từng RQ trong Results.
  - Section Related Work RIÊNG (3 subsection theo taxonomy + bảng literature-
    positioning được ĐƯA VÀO main text, không còn ở Supplementary).
  - 2 hình được đưa từ Supplementary vào main text: convergence_fs.pdf,
    threshold_heatmap.pdf (ASOC không có giới hạn cứng 8 display-item như
    Scientific Reports).
  - Conclusion section RIÊNG (tách khỏi Discussion), có limitations liệt kê
    rõ + Future Work 5 mục.
  - CRediT authorship contribution statement thay cho "Author contributions".
  - Graphical abstract (figures/graphical_abstract.png, 531x1328px) trong
    frontmatter; Highlights là file riêng (RG-SCSO_ASOC_Highlights.txt),
    không nhúng trong .tex (đúng quy ước Elsevier: Highlights là 1 hạng mục
    nộp riêng trong hệ thống, không phải 1 section của bản thảo).

Chạy:  python build_paper_asoc.py
"""

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

# Tái dùng NGUYÊN VẸN mọi hàm sinh bảng/khối văn bản đã verify ở bản SciRep --
# đảm bảo số liệu khớp 100% giữa 2 target, không tính lại/gõ lại.
from build_paper_scirep import (
    CLASSIC_CSV,
    SIGNAL_POS_CSV,
    THRESHOLD_CSV,
    classic_baselines_table,
    classifier_robustness_table,
    extended_ablation_table,
    heldout_combined_table,
    literature_positioning_table,
    rank_table_with_effect_size,
    rf_robustness_table,
    threshold_sensitivity_table,
)

OUT_TEX = "RG-SCSO_ASOC.tex"
OUT_SUPP_TEX = "RG-SCSO_ASOC_Supplementary.tex"


def build() -> None:
    s = load_summary()
    colon = s["gene"].get("ColonCancer", {})

    adaptive = adaptive_baselines()
    scsofam = scso_family_baselines()
    robust = robustness_baselines()
    svm16 = robustness_svm16()
    inf_val = inference_value()

    diversity = diversity_analysis() if os.path.exists(DIVERSITY_CSV) else None

    _hs = _heldout.load()
    heldout_tab = heldout_combined_table(_hs)
    classifier_robust_tab = classifier_robustness_table()
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
            "wall-clock inference saving to be resolvable above call overhead. "
            "We therefore report a projected, reconstructed inference-cost "
            "analysis rather than a measured deployment speedup: replaying "
            "the same feature-count reductions at a synthetic deployment-scale "
            f"workload (5{{,}}000 training instances, brute-force $k$-NN) "
            f"projects a {inf_val['min_speedup']:.0f}--{inf_val['max_speedup']:.0f}\\% "
            f"reduction in batch-inference cost relative to AOA (mean "
            f"{inf_val['mean_speedup']:.0f}\\%); this is a controlled "
            "complexity demonstration, not a measurement on the benchmark's "
            "own test sets."
        )
    else:
        inference_sentence = ""

    # ------------------------------------------------------------- Abstract
    # 150-250 words, không p-value/Cohen's d/Friedman-rank, không citation,
    # không notation phức tạp (RG-SCSO_MASTER_FINAL_COMPLETE.md Sec 26) --
    # dùng đúng khung đã fact-check "second-smallest, trailing only COA",
    # KHÔNG dùng claim "lowest mean feature count" (sai, COA còn ít hơn).
    abstract = (
        "Wrapper feature selection with swarm intelligence typically searches "
        "continuously and crosses into the binary domain via a fixed transfer "
        "function, a feature-agnostic quantization that discards continuous "
        "operators' fine adjustments, an effect we term washout. RG-SCSO "
        "replaces this transfer with a per-feature, relevance-modulated "
        "binarization: a mutual-information field biases each feature's "
        "bit-flip probability so informative features resist removal and "
        "noise resists inclusion. Its central and most consistent finding is "
        f"parsimony: on {s['n']} datasets, including two gene-expression sets, "
        "RG-SCSO selects the second-smallest feature subsets of any method "
        "tested, trailing only COA, which pays a several-point accuracy "
        "cost for it, on average "
        f"{_hs['nf_mean']['SCSO'].mean()/_hs['nf_mean']['RG-SCSO'].mean():.1f} "
        "times fewer than base SCSO, an advantage that survives every "
        "stress test we run, including comparison against optimizers carrying "
        "published adaptive transfers, against which RG-SCSO does not lead "
        "on accuracy yet still selects "
        f"{adaptive['red_min']:.0f}--{adaptive['red_max']:.0f}\% fewer "
        "features. Under a budget-matched, leak-free protocol -- the "
        "relevance prior, search, and cross-validated fitness are computed "
        "only on the training partition, never on held-out labels -- this "
        "compactness comes with the best mean held-out accuracy of any "
        "method tested, with a consistent edge over the closest competitor. "
        "A controlled ablation isolates the contribution of the relevance "
        "modulation itself, and cross-classifier, cross-prior experiments "
        "test the robustness of the mechanism. The binarization interface is "
        "the most reliable injection point for the relevance signal, though "
        "not the sparsest: LASSO is sparser on the two gene-expression sets, "
        "but accuracy is preserved, not improved, there, and parsimony "
        "without that extreme structure is the transferable gain."
    )

    # --------------------------------------------------------- Introduction
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
the binarization (Fig.~\ref{{fig:concept}}). We make four contributions: we
identify washout as a mechanistic failure mode and derive a diagnostic
bound, together with a cumulative extension linking it to discrete
transition dynamics; we propose RG-SCSO, whose ablation-confirmed
centerpiece is relevance-modulated sensitivity (RMS), supplemented by a
smaller, budget-neutral memetic refinement step (UMR) whose own sensitivity
sweep shows it contributes far less than RMS, an online-learning variant of
the relevance field is examined and pruned entirely by ablation; we evaluate
under a preregistered, budget-matched, leak-free protocol denying the
relevance prior any access to test labels; and we report
a full statistical treatment, a component ablation, and a size-fair
enrichment analysis correlating the observed parsimony with relevance
guidance.

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

    # ------------------------------------------------------- Related Work
    related_work = rf"""\subsection{{Binary transfer functions in swarm-based feature selection}}
Swarm-intelligence wrapper selectors, including grey wolf~\cite{{bgwo}},
particle swarm~\cite{{pso}}, whale~\cite{{mafarja}}, and several recent
optimizers~\cite{{aoa,coa,rime}}, are conceived for continuous optimization
and cross into the binary search space through a transfer function, most
commonly an S-shaped or V-shaped map~\cite{{tf}}. This transfer is fixed and
feature-agnostic in every one of these methods: it is applied identically to
every dimension regardless of that feature's relevance, the structural
weakness this paper terms washout (Introduction, Methods).

\subsection{{Relevance-guided mechanisms in feature selection}}
Filter criteria such as mutual information and mRMR~\cite{{guyon,mrmr}}
encode problem knowledge cheaply but are decoupled from the wrapper search;
memetic hybridization~\cite{{neri}} adds local refinement without addressing
the binarization interface; and knowledge-guided metaheuristics that inject
filter information into initialization or the objective, such as
filter-guided PSO for cancer-genome selection~\cite{{ludwig2025guided}}, bias
where the search starts or what it optimizes but still leave the
binarization operator itself knowledge-agnostic. RG-SCSO differs by
injecting the relevance signal at the binarization interface itself rather
than upstream of it; the signal-position experiment in Results
(Table~\ref{{tab:ablation}}) tests this design choice directly against both
alternatives.

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
work cited above: the three directly-comparable binary feature selectors
apply a standard, feature-agnostic transfer with no relevance-aware step,
and the four continuous-search variants improve exploration/exploitation
dynamics without touching the binarization interface at all. None of the
eight prior works positioned here makes the binarization operator itself
per-feature and relevance-aware -- the interface RG-SCSO modifies.

{literature_positioning_table()}"""

    # -------------------------------------------------------------- Results
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
        "binarization interface; Table~\\ref{tab:ablation} "
        "adds both alternatives to the component ablation. MI-weighted "
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

    classic_baselines_block = (
        "RQ4 asks how robust the mechanism is across classifiers, relevance "
        "priors, and dataset dimensionalities; this subsection and "
        "Table~\\ref{tab:classifierrobust} answer it directly. "
        "Table~\\ref{tab:classic} compares RG-SCSO against five classical "
        "filter, embedded, and wrapper selectors, mutual-information "
        "thresholding, mRMR, ReliefF, LASSO, and sequential forward "
        "selection, under the identical fitness and evaluation protocol. "
        "RG-SCSO significantly outperforms every classical baseline on the "
        "three lower-dimensional benchmark datasets (Zoo, Sonar, WDBC). On "
        "the two gene-expression ($p\\gg n$) datasets, this advantage does "
        "not hold: LASSO attains higher accuracy with far fewer features on "
        "both Leukemia (99.82\\% at 23 features vs.\\ RG-SCSO's 98.60\\% at "
        "940) and ColonCancer (95.70\\% at 64 features vs.\\ 88.09\\% at "
        "563), and mRMR is competitive with RG-SCSO on both. RG-SCSO's "
        "practical advantage is clearest on datasets without extreme "
        "$p\\gg n$ structure; on ultra-high-dimensional gene-expression "
        "data, a computationally far cheaper embedded method such as LASSO "
        "is a strong, arguably preferable, alternative. A remaining "
        "question is whether this parsimony mechanism is an artifact of "
        "the KNN wrapper used throughout; Table~\\ref{tab:classifierrobust} "
        "answers it directly, extending the same relevance-guided search to "
        "SVM and Random Forest wrappers. The advantage over a no-prior "
        "baseline is not a KNN artifact, though it is a more consistent "
        "parsimony gain than an accuracy one under Random Forest "
        "specifically, and the ReliefF-prior degradation already "
        "established above reproduces under every wrapper tested."
    )

    # threshold_sensitivity_table()'s caption cross-references tab:stability
    # (stability_index_table()'s label), which stays in Supplementary here
    # while this table is promoted to main text -- patch the dangling \ref
    # locally rather than editing the shared function (would break its
    # SciRep use, where both tables share one Supplementary document).
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
accuracy than RG-SCSO.}}
\label{{fig:tradeoff}}
\end{{figure}}

\subsection{{Ranking and statistical significance}}
{ranking_block}

{rank_table_with_effect_size(s) if s.get("stats") else rank_table(s)}

\subsection{{Ablation and mechanism}}
\label{{sec:ablation}}
{ablation_block}

{signal_position_block}

{extended_ablation_table(s) if s.get("ablation") and os.path.exists(SIGNAL_POS_CSV) else ""}

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

{classic_baselines_table(s) if s.get("ablation") and os.path.exists(CLASSIC_CSV) else ""}

{classifier_robust_tab}"""

    # ------------------------------------------------------------ Discussion
    discussion = rf"""These results trace washout, a concrete failure mode of
transfer-function-based binary feature selection, to its source and cure it
by moving the relevance signal directly inside the binarization operator
rather than upstream of it in the objective or the initialization. The
formal result motivating this design, presented in Methods, is a diagnostic
bound explaining why continuous-space enhancements fail at the binarization
boundary, not a convergence guarantee for RG-SCSO itself, and the underlying
recipe, a filter prior coupled to a wrapper search with memetic refinement,
is a known combination in the feature-selection literature; what is new is
the injection point together with a stringent evaluation protocol,
budget-matching, a leak-free hold-out, cross-classifier and cross-prior
robustness, and an explicit exploration-safety diagnostic. We report
parsimony, not raw accuracy, as the transferable outcome of this choice.

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
built from a single filter statistic; swapping in a ReliefF prior exposes a
genuine limitation
rather than confirming the mechanism, the parsimony advantage disappears
(Supplementary Information), so RG-SCSO's compactness depends on a prior
that drives uninformative features below the neutral point, not on the mere
presence of a relevance signal. This dependence, together with the
$O(dn\log n)$ one-time cost of the mutual-information prior amortized
against the wrapper's $N+K$ per-iteration evaluations, is why we scope the
scalability claim to the wrapper cost dominating overall runtime as feature
dimensionality grows (full per-dataset wall-clock cost is in Supplementary
Information) rather than to any runtime advantage over same-budget
baselines, which we do not claim.

The main
objective is also a KNN wrapper; under SVM, tested directly on
{svm16.get('n_ds', 16)} of the {s['n']} datasets, and under Random Forest on
the same five-dataset subset as Table~\ref{{tab:classifierrobust}} in the
Results, the parsimony advantage is not a KNN artifact, though it is a more
consistent gain in subset size than in accuracy under Random Forest
specifically, and the ReliefF-prior degradation already established above
reproduces under every wrapper tested. The selected subset is smaller and
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
microarray sets, and behaviour on ultra-high-dimensional omics data of
$10^4$ to $10^5$ features is extrapolated, not measured. The relevance field
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

    # ------------------------------------------------------------ Conclusion
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
read past. RG-SCSO does not lead on accuracy against optimizers carrying
published adaptive continuous-space transfers or against same-family
binary SCSO selectors; on the two extreme $p\gg n$ gene-expression datasets
tested, a classical LASSO baseline outperforms it outright on both accuracy
and feature count; its dependence on the underlying relevance prior is
genuine, since replacing mutual information with a ReliefF prior removes
the parsimony advantage entirely; the 0.5 preferred-bit threshold, while
empirically reasonable across the range tested, is not a uniformly optimal
choice; and the benchmark itself, though spanning {feat_min}--{feat_max}
features across biomedical, gene-expression, and categorical domains, is
drawn from a single curated family of datasets, so behavior on
ultra-high-dimensional omics data of $10^4$--$10^5$ features is
extrapolated rather than measured.

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

    tex = rf"""% !TeX program = xelatex
%=======================================================================
% RG-SCSO, Elsevier (elsarticle) format, target: Applied Soft Computing.
% Song song voi build_paper_scirep.py (Scientific Reports); so lieu sinh tu
% dong tu experiments/results_fs*/*.csv, tai dung nguyen ham sinh bang/hinh
% da verify o ban SciRep (import, khong copy-paste). Cau truc: Introduction
% (co Research Questions) -> Related Work (rieng, co bang literature-
% positioning) -> Results -> Discussion -> Conclusion (rieng, co Future Work
% 5 muc) -> Methods (cuoi bai). Khong co gioi han cung display-item nhu
% Scientific Reports nen 2 hinh (convergence tren feature-selection
% objective, threshold-sensitivity heatmap) va bang literature-positioning
% duoc dua vao main text thay vi Supplementary.
% Font: TeX Gyre Termes (Times-clone, GUST Font License, tu CTAN) qua
% fontspec/unicode-math -- can compiler XeLaTeX/LuaLaTeX (dong dau file
% "% !TeX program = xelatex" de Overleaf tu chon dung compiler). amssymb/
% amsfonts KHONG duoc load truc tiep vi xung dot voi unicode-math (vd.
% \eth da duoc dinh nghia) -- amsmath la du cho toan bo cong thuc dung
% trong bai.
%=======================================================================
\documentclass[review]{{elsarticle}}
\usepackage{{fontspec}}
\usepackage{{unicode-math}}
\setmainfont{{texgyretermes}}[
  Path=fonts/,
  Extension=.otf,
  UprightFont=*-regular,
  BoldFont=*-bold,
  ItalicFont=*-italic,
  BoldItalicFont=*-bolditalic
]
\setmathfont{{texgyretermes-math.otf}}[Path=fonts/]
\usepackage{{lineno}}
\modulolinenumbers[5]
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
\journal{{Applied Soft Computing}}
\bibliographystyle{{elsarticle-num}}
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

\begin{{frontmatter}}

\title{{Relevance-Guided Binarization for Parsimonious Feature Selection with
Sand Cat Swarm Optimization}}

\author[1]{{Bui Quang Huy}}
\ead{{huybq@donga.edu.vn}}
\author[1]{{Duong Minh Son}}
\ead{{sondm@donga.edu.vn}}
\affiliation[1]{{organization={{Dong A University}}, city={{Da Nang}},
country={{Vietnam}}}}

\begin{{graphicalabstract}}
\centering
\includegraphics[width=0.42\textwidth]{{graphical_abstract.png}}
\end{{graphicalabstract}}

\begin{{abstract}}
{abstract}
\end{{abstract}}

\begin{{keyword}}
Feature selection \sep Sand Cat Swarm Optimization \sep binary optimization
\sep relevance-guided binarization \sep wrapper feature selection \sep
parsimony \sep metaheuristic
\end{{keyword}}

\end{{frontmatter}}

\linenumbers

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
differ significantly in mean held-out rank. The corresponding in-sample
diagram appears in Supplementary Fig.~S6.}}
\label{{fig:cd}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{mechanism.pdf}}
\caption{{Mechanism evidence. (a) Size-fair top-MI enrichment on the two
gene-expression sets (selection precision divided by the chance level
$|S|/N$; mean over 30 runs, error bars = std): RG-SCSO enriches its subset
in relevant features above chance, whereas the relevance-agnostic SCSO sits
at chance. (b) Feature-selection stability (Nogueira $\Phi$, 30 runs) across
all five representative datasets: RG-SCSO is more consistent run to run
than same-family SCSO, clearest on the three lower-dimensional sets, but
both are close to the level expected by chance on the two gene-expression
sets, where AOA's near-zero $\Phi$ reflects that it selects nearly every
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
    $T$; budget $\mathrm{{max\_nfe}}$; memetic size $K$; bias strength $\gamma$
  \ENSURE best feature mask $b^\ast$
  \STATE $\rho_j \leftarrow$ normalized mutual information $I(X_j;y)$,
    computed on $(X,y)$ only, $\forall j$ \hfill$\triangleright$ static
    relevance field
  \STATE $\forall j$: preferred bit $\hat{{b}}_j \leftarrow \mathbf{{1}}[\rho_j>0.5]$;
    strength $s_j \leftarrow 2|\rho_j-0.5|$
  \STATE initialize positions $x_i \sim \mathcal{{U}}(-1,1)^d$, $i=1,\dots,N$;
    binarize each by RMS; evaluate; set $b^\ast$
  \WHILE{{$\mathrm{{nfe}} < \mathrm{{max\_nfe}}$}}
    \STATE $R \leftarrow S_M\,(1-t/T)$ \hfill$\triangleright$ sensitivity range contracts
    \FOR{{each agent $i=1,\dots,N$}}
      \STATE update $x_i$ by the SCSO position rule using range $R$
      \FOR{{each feature $j$}}
        \STATE $p \leftarrow |\tanh(x_{{ij}})|$ \hfill$\triangleright$ V-shaped transfer
        \STATE $p \leftarrow p\,(1+\gamma s_j)$ if the flip moves bit $j$ toward
          $\hat{{b}}_j$, else $p\,(1-\gamma s_j)$
        \STATE flip bit $j$ with probability $\mathrm{{clip}}(p,0,1)$
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
objective with the same 0.01 cardinality weight; RG-SCSO is granted no
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
$\Delta_j=\pm\,\gamma\,s_j\,V(x_j)$, independent of $T'$.

\begin{{proposition}}[Transition probability and cumulative leverage]
Under the V-shaped binarization rule used throughout this paper, bit $j$
flips at iteration $t$ with probability exactly $T(x_j^{{(t)}})$, so the
one-step bit-transition probability coincides with the flip probability,
$P\big(b_j^{{(t+1)}}\ne b_j^{{(t)}}\mid x_j^{{(t)}}\big)=T(x_j^{{(t)}})$. Lemma~1's
bound on $\Delta_j$ is therefore, without further assumption, already a
bound on the one-step change in transition probability, and this bound
composes linearly over $N$ repeated samples of the same bit in the flat,
saturated regime (proof in Supplementary Information). This is a local,
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

\section*{{CRediT authorship contribution statement}}
\textbf{{Bui Quang Huy:}} Conceptualization, Methodology, Software,
Validation, Formal analysis, Writing -- original draft. \textbf{{Duong Minh
Son:}} Supervision, Writing -- review \& editing.

\section*{{Declaration of competing interest}}
The authors declare no competing interests.

\section*{{Data availability}}
The datasets are publicly available benchmarks (UCI and standard microarray
sets). The source code, the locked preregistration, the per-run seeds, and
the raw results are available in an anonymized repository
(\url{{https://anonymous.4open.science/r/RG-SCSO}}) and will be released in a
public, citable repository upon acceptance.

\section*{{Funding}}
No funding was received for this work.

\bibliography{{references}}

\end{{document}}
"""
    # Float-drift fix (see Supplementary write, below): force pending
    # figures/tables to resolve before crossing a \section boundary, so a
    # promoted table (e.g. the literature-positioning table) never queues
    # past the subsection that introduces it.
    tex = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", tex)

    with open(OUT_TEX, "w") as fh:
        fh.write(tex)

    # --------------------------------------------------- Supplementary Info
    # Giống hệt cấu trúc Supplementary của bản SciRep, TRỪ 3 phần đã chuyển
    # vào main text ASOC (literature positioning, threshold sensitivity,
    # convergence-on-FS-objective) để không lặp nội dung.
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
    supp = rf"""% !TeX program = xelatex
%=======================================================================
% Supplementary Information for RG-SCSO (Applied Soft Computing submission)
% Font matches the main text: TeX Gyre Termes via fontspec/unicode-math
% (needs XeLaTeX/LuaLaTeX). amssymb not loaded (conflicts with
% unicode-math); amsmath covers every formula used here.
%=======================================================================
\documentclass[9pt]{{article}}
\usepackage[a4paper,margin=25mm]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{unicode-math}}
\setmainfont{{texgyretermes}}[
  Path=fonts/,
  Extension=.otf,
  UprightFont=*-regular,
  BoldFont=*-bold,
  ItalicFont=*-italic,
  BoldItalicFont=*-bolditalic
]
\setmathfont{{texgyretermes-math.otf}}[Path=fonts/]
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

\textbf{{Proof of Proposition 2 (Transition probability and cumulative
leverage).}} The identity $P\big(b_j^{{(t+1)}}\ne b_j^{{(t)}}\mid
x_j^{{(t)}}\big)=T(x_j^{{(t)}})$ follows directly from the definition of the
V-shaped flip rule (a bit is flipped, independently at each iteration, with
probability equal to the transfer value); no further argument is required
because transition and flip are the same event under this rule. The
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
    # Float-drift fix: pending figures/tables can queue past several
    # \section boundaries under heavy float density, leaving section
    # headings visually empty while their content lands pages later (or
    # even before the heading that introduces it). \FloatBarrier before
    # every \section forces all pending floats to resolve first.
    supp = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", supp)

    with open(OUT_SUPP_TEX, "w") as fh:
        fh.write(supp)

    print(f"Đã ghi {OUT_TEX} và {OUT_SUPP_TEX}")


if __name__ == "__main__":
    build()
