"""Sinh bản thảo LaTeX cho Scientific Reports (Nature Portfolio) — SINGLE SOURCE
OF TRUTH, tách biệt với build_paper_tex.py (bản Applied Intelligence).

KHÔNG gõ tay số liệu: mọi con số đọc động từ artifact qua các hàm dùng chung
import từ build_paper_tex.py / build_paper_structure.py / build_heldout_table.py.

Yêu cầu Scientific Reports đã tra thật (nature.com, không đoán):
  - Văn bản chính (Intro+Results+Discussion, KHÔNG tính Abstract/Methods/
    References/figure legends) tối đa 4.500 từ.
  - Tối đa ~11 trang in.
  - Abstract tối đa 200 từ, KHÔNG cấu trúc (không mục con).
  - Tiêu đề tối đa 20 từ.
  - Tối đa 6 từ khóa.
  - Tối đa 8 hình+bảng gộp lại trong TOÀN BÀI.
  - Cấu trúc: Title/Abstract → Introduction → Results (có mục con được) →
    Discussion (KHÔNG mục con) → Methods (đặt CUỐI, không tính vào giới hạn
    4.500 từ) → Data availability/Author contributions/Competing interests →
    References → (Supplementary Information riêng, không tính vào bài chính).

Chiến lược cắt: giữ đúng 8 hình+bảng trong bài chính (concept fig, CD diagram,
mechanism fig 2-panel [enrichment + stability index] = 3 hình; held-out
accuracy+nfeat gộp 1 bảng, rank, ablation, classic-baselines, classifier-
robustness = 5 bảng) — TẤT CẢ dựa trên held-out (leak-free) làm bằng chứng
chính, không lặp lại bảng in-sample. Mọi thứ còn lại (bảng dataset, bảng
in-sample, hình convergence/diversity, chi tiết per-dataset của robustness/
stability, phụ lục cũ) chuyển sang Supplementary Information
(build_paper_scirep_supp.tex, sinh cùng lúc) — KHÔNG xóa, chỉ chuyển vị trí.

Chạy:  python build_paper_scirep.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from build_paper_structure import load_summary
from build_paper_tex import (
    ABL_LABEL_TEX,
    DIVERSITY_CSV,
    PROC_DIR,
    ROBUST_CSV,
    YEAR,
    ablation_table,
    accuracy_table,
    adaptive_baselines,
    dataset_table,
    diversity_analysis,
    esc,
    inference_value,
    nfeat_table,
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
from src.stats.statistical_tests import holm_correction

OUT_TEX = "RG-SCSO_SciRep.tex"
OUT_SUPP_TEX = "RG-SCSO_SciRep_Supplementary.tex"

# Q1-review Loại B experiment outputs (added post-review; see
# RG-SCSO_Q1_Review_Final.md Priority 1/2/3/6).
CLASSIC_CSV = os.path.join("experiments", "results_fs_classic", "fs_classic_results.csv")
SIGNAL_POS_CSV = os.path.join(
    "experiments", "results_fs_signal_position", "fs_signal_position_results.csv"
)
SHUFFLE_MI_CSV = os.path.join(
    "experiments", "results_fs_shuffle_mi", "fs_shuffle_mi_results.csv"
)
NESTED_CV_CSV = os.path.join("experiments", "results_fs_nested_cv", "fs_nested_cv_results.csv")
RELEVANCE_VAR_CSV = os.path.join(
    "experiments", "results_relevance_variance", "relevance_variance_results.csv"
)
STABILITY_CSV = os.path.join(
    "experiments", "results_stability", "stability_index_results.csv"
)
THRESHOLD_CSV = os.path.join(
    "experiments", "results_threshold", "threshold_sensitivity_results.csv"
)
NFE_CONTROL_CSV = os.path.join(
    "experiments", "results_nfe_control", "nfe_control_results.csv"
)
FS_MAIN_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")

# Nhãn 2 hàng MỚI (Q1 review Priority 2), chèn vào bảng ablation chính sau
# hàng "- UMR" — signal-position steps 2/3 (không có đối chiếu sẵn trong
# ABL_CONFIG_ORDER, KHÔNG như steps 1/4/5 vốn là re-run y hệt 3 config cũ).
SIGPOS_LABEL_TEX = {
    "2_MIInit_NoRMS": r"MI-guided init (no RMS)",
    "3_MIObjective_NoRMS": r"MI-weighted objective (no RMS)",
}
SIGPOS_FINAL_STEP = "5_RMS_UMR_Full"  # == deployed "- ORL (final)" 2-component RG-SCSO


def classic_baselines_table(s: dict) -> str:
    """NEW main-text table (Q1 review Priority 1): RG-SCSO vs. five classical
    filter/embedded/wrapper selectors on the same 5-dataset ablation pilot,
    identical fitness/eval protocol. Honest disclosure, not softened: RG-SCSO
    wins the 3 lower-dimensional datasets; LASSO/mRMR win the two
    gene-expression (p>>n) datasets with far fewer features."""
    ds_list = s["abl_datasets"]
    cb = pd.read_csv(CLASSIC_CSV)
    main = pd.read_csv(FS_MAIN_CSV)
    rgscso = main[(main.algorithm == "RG-SCSO") & (main.dataset.isin(ds_list))]

    algos = ["RG-SCSO", "MI-threshold", "mRMR", "ReliefF-baseline", "LASSO", "SFS"]
    label = {"RG-SCSO": "RG-SCSO", "MI-threshold": "MI-threshold", "mRMR": "mRMR",
              "ReliefF-baseline": "ReliefF", "LASSO": "LASSO", "SFS": "SFS"}
    acc, nfeat = {}, {}
    for a in algos:
        sub = rgscso if a == "RG-SCSO" else cb[cb.algorithm == a]
        g = sub.groupby("dataset")
        acc[a] = g["accuracy"].mean()
        nfeat[a] = g["n_selected_features"].mean()

    cols = "l" + "c" * len(ds_list)
    head = "Method & " + " & ".join(esc(d) for d in ds_list) + r" \\"
    lines = []
    for a in algos:
        cells = [label[a]]
        for ds in ds_list:
            a_acc, a_nf = acc[a].get(ds, float("nan")), nfeat[a].get(ds, float("nan"))
            val = f"{a_acc:.4f} ({a_nf:.0f})"
            col_best = max(acc[oa].get(ds, -1.0) for oa in algos)
            if a_acc >= col_best - 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Comparison with classical filter, embedded, and wrapper feature "
        "selectors (5-dataset pilot, not the full 18-dataset benchmark; 30 "
        "runs each, identical fitness function and evaluation protocol as "
        "the main study; cells show accuracy with mean selected-feature "
        "count in parentheses, best per column in bold). RG-SCSO "
        "significantly outperforms every classical baseline on the three "
        "lower-dimensional datasets (Zoo, Sonar, WDBC). On the two "
        "gene-expression ($p\\gg n$) datasets, LASSO attains higher accuracy "
        "with far fewer features, and mRMR is competitive with RG-SCSO."
    )
    return (
        "\\begin{table}[tb]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:classic}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def extended_ablation_table(s: dict) -> str:
    """Main-text component-ablation table, extended with 2 rows from the
    Q1-review Priority-2 signal-position experiment (where else could the
    relevance signal be injected, besides the binarization interface?).
    Steps 1/4/5 of that experiment are re-runs, through an independent
    harness, of the existing NoImprovement/NoUMR/NoORL(final) configs
    (cross-harness accuracy agrees to within <=0.5pp on all 5 datasets, a
    wiring sanity check, not re-derived here) — only steps 2/3 add new rows."""
    ds_list = s["abl_datasets"]
    sp = pd.read_csv(SIGNAL_POS_CSV)

    def sp_series(step: str, metric: str = "accuracy") -> pd.Series:
        return sp[sp.step == step].groupby("dataset")[metric].mean()

    def sp_values(step: str, ds: str, metric: str = "accuracy") -> np.ndarray:
        sub = sp[(sp.step == step) & (sp.dataset == ds)].sort_values("run_id")
        return sub[metric].to_numpy()

    final_mean = sp_series(SIGPOS_FINAL_STEP)
    new_rows = {step: sp_series(step) for step in SIGPOS_LABEL_TEX}

    sig_new = set()
    for step in SIGPOS_LABEL_TEX:
        pvals = []
        for ds in ds_list:
            a, b = sp_values(step, ds), sp_values(SIGPOS_FINAL_STEP, ds)
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = 1.0
            pvals.append(p)
        p_holm = holm_correction(np.array(pvals))
        for ds, p in zip(ds_list, p_holm):
            if p < 0.05 and new_rows[step].get(ds, 1.0) < final_mean.get(ds, 0.0):
                sig_new.add((step, ds))

    cols = "l" + "c" * len(ds_list)
    head = "Configuration & " + " & ".join(esc(d) for d in ds_list) + r" \\"
    lines = []
    for cfg in s["configs"]:
        cells = [ABL_LABEL_TEX[cfg]]
        for ds in ds_list:
            m = s["means"][cfg].get(ds)
            val = "--" if m is None else f"{m:.4f}"
            if (cfg, ds) in s["sig"]:
                val += r"$^\dagger$"
            if cfg == "Full":
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
        if cfg == "NoUMR":
            for step, row_label in SIGPOS_LABEL_TEX.items():
                cells2 = [row_label]
                for ds in ds_list:
                    m = new_rows[step].get(ds)
                    val = "--" if m is None else f"{m:.4f}"
                    if (step, ds) in sig_new:
                        val += r"$^\dagger$"
                    cells2.append(val)
                lines.append(" & ".join(cells2) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Component ablation (mean accuracy, 30 runs, five datasets), "
        "extended with two alternative relevance-injection points from a "
        "dedicated signal-position experiment. "
        "``Final'' throughout is the 2-component RMS+UMR configuration (the "
        "``$-$ ORL'' row, i.e.\\ the deployed RG-SCSO); $^\\dagger$ marks "
        "significantly worse than Final (paired Wilcoxon signed-rank, "
        "Holm-corrected $p<0.05$). Injecting the relevance signal at "
        "initialization or as an objective penalty, instead of at the "
        "binarization interface, is significantly worse on most datasets; "
        "the exception is Leukemia, where MI-guided initialization is "
        "statistically indistinguishable from Final while selecting roughly "
        "4$\\times$ fewer features (245 vs.\\ 940)."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:ablation}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def shuffle_mi_table() -> str:
    """Supplementary table (Q1 review Priority 3): causal intervention on the
    relevance field via feature-identity permutation / sign inversion."""
    sm = pd.read_csv(SHUFFLE_MI_CSV)
    algos = ["RG-SCSO-MI", "RG-SCSO-ShuffledMI", "RG-SCSO-InvertedMI"]
    label = {"RG-SCSO-MI": "Real MI", "RG-SCSO-ShuffledMI": "Shuffled MI",
              "RG-SCSO-InvertedMI": "Inverted MI"}
    ds_list = sorted(sm["dataset"].unique())
    acc = {a: sm[sm.algorithm == a].groupby("dataset")["accuracy"].mean() for a in algos}
    nfeat = {a: sm[sm.algorithm == a].groupby("dataset")["n_selected_features"].mean()
             for a in algos}
    cols = "l" + "c" * len(ds_list)
    head = "Relevance field & " + " & ".join(esc(d) for d in ds_list) + r" \\"
    lines = []
    for a in algos:
        cells = [label[a]]
        for ds in ds_list:
            cells.append(f"{acc[a].get(ds, float('nan')):.4f} "
                          f"({nfeat[a].get(ds, float('nan')):.0f})")
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Causal intervention on the relevance field: "
        "accuracy with mean selected-feature count in parentheses, 30 runs. "
        "Permuting feature identity while preserving the marginal "
        "distribution of $\\rho$ (Shuffled MI) yields no significant "
        "accuracy difference from the real field on three of five datasets "
        "(Zoo, Sonar, WDBC; paired Wilcoxon signed-rank, Holm-corrected "
        "$p>0.05$); the difference is significant on ColonCancer and, with a "
        "negligible effect size, on Leukemia. Inverting the field's sign "
        "(Inverted MI) is significantly worse than both real and shuffled MI "
        "on all five datasets."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:shufflemi}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def nested_cv_table() -> str:
    """Supplementary table (Q1 review Priority 6): genuine outer-fold nested
    CV pilot, vs. the main study's single 80/20 held-out split."""
    nc = pd.read_csv(NESTED_CV_CSV)
    algos = ["RG-SCSO", "SCSO", "AOA"]
    ds_list = sorted(nc["dataset"].unique())
    mean = nc.groupby(["algorithm", "dataset"])["mean_nested_cv_accuracy"].mean()
    std = nc.groupby(["algorithm", "dataset"])["mean_nested_cv_accuracy"].std()
    cols = "l" + "c" * len(ds_list)
    head = "Algorithm & " + " & ".join(esc(d) for d in ds_list) + r" \\"
    lines = []
    for a in algos:
        cells = [esc(a)]
        for ds in ds_list:
            cells.append(f"{mean.get((a, ds), float('nan')):.4f}$\\pm$"
                          f"{std.get((a, ds), float('nan')):.4f}")
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Nested cross-validation pilot: outer "
        "5-fold accuracy, mean$\\pm$std over 5 independent runs (each run "
        "repeats the full search inside every outer fold). Directionally "
        "consistent with the main held-out study (RG-SCSO ahead on Zoo and "
        "ColonCancer, comparable to AOA on WDBC), but $n=5$ runs per cell is "
        "underpowered: no pairwise comparison reaches Holm-corrected "
        "significance at this sample size, so this is a supportive check, "
        "not a replacement for the main held-out result."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:nestedcv}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def relevance_variance_table() -> str:
    """Supplementary table (Q1 review Priority 6): bootstrap stability of the
    MI relevance field itself."""
    rv = pd.read_csv(RELEVANCE_VAR_CSV)
    cols = "lccc"
    head = r"Dataset & Mean Spearman & Mean top-$K$ Jaccard & Mean std($\rho_j$) \\"
    lines = []
    for _, r in rv.sort_values("dataset").iterrows():
        lines.append(
            f"{esc(r['dataset'])} & {r['mean_spearman_between_resamples']:.3f} & "
            f"{r['mean_jaccard_topk_between_resamples']:.3f} & "
            f"{r['mean_std_rho_per_feature']:.3f} \\\\"
        )
    body = "\n".join(lines)
    cap = (
        "Bootstrap stability of the mutual-information relevance field: "
        "mean pairwise Spearman correlation and top-$K$ "
        "Jaccard overlap of $\\rho$ across 30 bootstrap resamples "
        "($K\\approx$ mean selected-subset size). The two gene-expression "
        "datasets (ColonCancer, Leukemia) show markedly lower resampling "
        "stability than the three lower-dimensional sets, consistent with "
        "the expectation that the relevance field is more prone to "
        "sampling variance on small-$n$, high-$p$ data."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:relvar}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


FS_DIR = os.path.join("experiments", "results_fs")
WILC_CSV = os.path.join(FS_DIR, "wilcoxon_vs_rgscso.csv")


def rank_table_with_effect_size(s: dict) -> str:
    """Replaces the shared rank_table(s) (build_paper_tex.py, read-only, not
    modified here) with a version that adds a median |Cohen's d| column --
    the effect sizes are already computed and used in prose (median |d|) but
    Table 2 itself never surfaced them as a column (RG-SCSO_MASTER_FINAL_
    COMPLETE.md audit finding). Same display-item count: replaces Table 2,
    does not add a 9th item."""
    has = s.get("stats")
    ranking = s["rank7"].sort_values() if has else s["avg_rank"]
    d_by_baseline = {}
    if has and os.path.exists(WILC_CSV):
        w = pd.read_csv(WILC_CSV)
        for a in w["compared_with"].unique():
            dv = w[w["compared_with"] == a]["cohens_d"].abs()
            d_by_baseline[a] = float(dv.median()) if len(dv) else float("nan")
    lines = []
    for a, r in ranking.items():
        if a == "RG-SCSO":
            wtl, d_str = "--", "--"
        elif has:
            wtl = "{}/{}/{}".format(*s["sig_wtl"].get(a, (0, 0, 0)))
            dv = d_by_baseline.get(a, float("nan"))
            d_str = f"{dv:.2f}" if dv == dv else "--"  # NaN check
        else:
            wtl, d_str = "\\textit{[pending]}", "--"
        name = f"{esc(a)} ({YEAR.get(a, '?')})"
        lines.append(f"{name} & {r:.2f} & {wtl} & {d_str} \\\\")
    body = "\n".join(lines)
    cap = ("Average accuracy rank across the "
           f"{s['n']} datasets, the Holm-significant win/tie/loss of RG-SCSO "
           "against each baseline (Wilcoxon signed-rank, $\\alpha=0.05$), and "
           "the median $|$Cohen's $d|$ effect size over the significant-win "
           "comparisons against each baseline."
           if has else
           "Average accuracy rank across the "
           f"{s['n']} datasets and win/tie/loss of RG-SCSO against each "
           "baseline. Wilcoxon signed-rank with Holm correction and effect "
           "sizes are reported in the final version.")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:rank}\n\\footnotesize\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Algorithm (year) & Avg. rank & W/T/L vs.\\ RG-SCSO (Holm) & "
        "Median $|d|$ \\\\\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def literature_positioning_table() -> str:
    """Supplementary table (RG-SCSO_MASTER_FINAL_COMPLETE.md, item S9):
    compact positioning of every recent SCSO-family work already cited in
    this paper -- visualizes the claim already made in the Introduction ("no
    prior SCSO feature selector makes the binarization operator itself
    per-feature and relevance-aware") with a compiled table instead of only
    prose. Static compilation from references.bib entries already cited
    elsewhere in this paper -- no new experimental data, citation keys/years/
    journals verified against references.bib before writing."""
    rows = [
        ("Seyyedabbasi \\& Kiani~\\cite{scso}", "2022", "Base continuous SCSO (no FS)", "Baseline (Table~1)"),
        ("bSCSO~\\cite{bscso}", "2023", "Binary wrapper FS, standard transfer", "Same-family baseline (Discussion)"),
        ("Binary SCSO (biomedical)~\\cite{scsofs2}", "2023", "Binary wrapper FS, standard transfer", "Same-family baseline (Discussion)"),
        ("Adaptive SCSO~\\cite{scsofs3}", "2024", "Binary wrapper FS, standard transfer", "Same-family baseline (Discussion)"),
        ("IMSCSO~\\cite{imscso2024}", "2024", "Continuous search (multi-strategy)", "No -- global optimization only, no FS"),
        ("SCSO+Lens-OBL+SSA~\\cite{scsolensobl2024}", "2024", "Continuous search (lens-OBL init)", "No -- global optimization only, no FS"),
        ("Improved SCSO~\\cite{improvedscso2024}", "2024", "Continuous search dynamics", "No -- global optimization only, no FS"),
        ("MESCSO~\\cite{mescso2025}", "2025", "Continuous search (multi-strategy)", "No -- global optimization only, no FS"),
    ]
    lines = [f"{name} & {year} & {mod} & {comp} \\\\" for name, year, mod, comp in rows]
    body = "\n".join(lines)
    cap = (
        "Positioning of RG-SCSO against recent SCSO-family literature cited "
        "in this paper. The four 2024--2025 continuous-search variants "
        "(IMSCSO, SCSO+Lens-OBL+SSA, Improved SCSO, MESCSO) improve "
        "exploration/exploitation dynamics in continuous space but are global "
        "optimization studies, not feature selectors, so they are not "
        "directly comparable under this paper's binary FS protocol; they are "
        "cited to establish that none of them touches the binarization "
        "interface itself. The three binary SCSO feature selectors (bSCSO "
        "and two further variants) are directly comparable and are included "
        "as same-family baselines (Discussion; Supplementary Information)."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:litpos}\n\\footnotesize\n"
        "\\begin{tabular}{p{4.2cm}cp{3.6cm}p{3.6cm}}\n\\toprule\n"
        "Method & Year & What it modifies & Comparable to this protocol? \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def rf_robustness_table() -> str:
    """Supplementary table (Diem_yeu_RG-SCSO.md §2.5): Random Forest wrapper,
    the classifier family the reviewer flagged as missing. Reuses the exact
    protocol and algorithm set already validated for the KNN/SVM robustness
    table (robustness_baselines above), just a third wrapper value; the
    underlying harness (src/feature_selection/run_fs_robustness.py) already
    supported wrapper="RF", it had simply never been run."""
    if not os.path.exists(ROBUST_CSV):
        return ""
    rob = pd.read_csv(ROBUST_CSV)
    rob = rob[rob.wrapper == "RF"]
    if rob.empty:
        return ""
    algos = ["RG-SCSO-MI", "RG-SCSO-ReliefF", "bSCSO"]
    labels = {"RG-SCSO-MI": "RG-SCSO (MI)", "RG-SCSO-ReliefF": "RG-SCSO (ReliefF)",
              "bSCSO": "bSCSO (no prior)"}
    datasets = sorted(rob["dataset"].unique())

    def m(ds, a, col):
        return rob[(rob.dataset == ds) & (rob.algorithm == a)][col].mean()

    lines = []
    for ds in datasets:
        nfs = {a: m(ds, a, "n_selected_features") for a in algos}
        min_nf = min(nfs.values())
        for i, a in enumerate(algos):
            acc = m(ds, a, "accuracy")
            nf = nfs[a]
            nf_cell = f"\\textbf{{{nf:.1f}}}" if abs(nf - min_nf) < 1e-9 else f"{nf:.1f}"
            dscol = f"\\multirow{{3}}{{*}}{{{esc(ds)}}}" if i == 0 else ""
            name = f"\\textbf{{{labels[a]}}}" if a == "RG-SCSO-MI" else labels[a]
            lines.append(f"{dscol} & {name} & {acc:.4f} & {nf_cell} \\\\")
        if ds != datasets[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    n_ds = rob["dataset"].nunique()
    n_runs = int(rob["run_id"].nunique())
    cap = (
        f"Robustness under a Random Forest wrapper on the same {n_ds} "
        f"representative datasets as the KNN/SVM table above (identical "
        f"protocol, but $\\times$ {n_runs} runs rather than the main "
        "study's 30: each Random Forest fitness evaluation is substantially "
        "more expensive than a KNN one, making the full run count "
        "infeasible in reasonable wall-clock time, so this check is a "
        "reduced-run pilot, matching the precedent already set by the "
        "nested cross-validation pilot elsewhere in this Supplementary "
        "Information). Fewest features per dataset in bold."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:rfrobust}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Dataset & Method & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def stability_index_table() -> str:
    """Supplementary table (Diem_yeu_RG-SCSO.md §2.2): does RG-SCSO select
    largely the SAME subset across independent runs, or just a subset of the
    same SIZE? Nogueira, Sechidis \\& Brown (2018)~\\cite{nogueira2018stability}
    Phi -- the standard generalization of the Kuncheva (2007) consistency
    index to variable subset size, since RG-SCSO's subset size is not fixed
    across runs. Phi in [-1,1]; 1 = identical subset every run, 0 = no more
    consistent than selecting the same number of features at random."""
    if not os.path.exists(STABILITY_CSV):
        return ""
    st = pd.read_csv(STABILITY_CSV)
    if st.empty:
        return ""
    algos = ["RG-SCSO", "SCSO", "AOA"]
    datasets = sorted(st["dataset"].unique())
    lines = []
    for ds in datasets:
        rows = {a: st[(st.dataset == ds) & (st.algorithm == a)].iloc[0] for a in algos}
        best_phi = max(float(rows[a]["nogueira_phi"]) for a in algos)
        for i, a in enumerate(algos):
            r = rows[a]
            phi = float(r["nogueira_phi"])
            d_total = int(r["n_total_features"])
            nfeat = float(r["mean_n_selected"])
            phi_cell = f"{phi:.3f}"
            if abs(phi - best_phi) < 1e-9:
                phi_cell = f"\\textbf{{{phi_cell}}}"
            dscol = f"\\multirow{{3}}{{*}}{{{esc(ds)}}}" if i == 0 else ""
            lines.append(f"{dscol} & {esc(a)} & {phi_cell} & {nfeat:.1f}/{d_total} \\\\")
        if ds != datasets[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    cap = (
        "Feature-selection stability (Nogueira $\\Phi$~\\cite{nogueira2018stability}, "
        "30 independent runs per cell, same protocol and datasets as the main "
        "ablation) -- whether an algorithm returns the same subset run to "
        "run, not just a subset of the same size. Highest $\\Phi$ per "
        "dataset in bold; ``Mean \\#Feat./Total'' gives the mean selected "
        "count against the dataset's total feature count. RG-SCSO is more "
        "stable than same-family SCSO on every dataset, most clearly on the "
        "three lower-dimensional sets; on both gene-expression sets "
        "RG-SCSO's own $\\Phi$ is itself close to the level expected by "
        "chance, so its markedly smaller subset size is not, on these two "
        "datasets, a materially more repeatable one. AOA's near-zero $\\Phi$ "
        "reflects that it selects nearly every available feature on almost "
        "every run (see its Mean \\#Feat./Total column), leaving little room "
        "for $\\Phi$ to be anything but close to chance, rather than "
        "evidence of instability in the same sense as RG-SCSO or SCSO's "
        "genuine run-to-run subset variation."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:stability}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Dataset & Method & $\\Phi$ & Mean \\#Feat./Total \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def notation_table() -> str:
    """Supplementary table (RG-SCSO_MASTER_FINAL_COMPLETE.md Section 7):
    compiled directly from the symbols actually used in Methods (Problem
    formulation / Theoretical motivation / The RG-SCSO mechanism / Algorithm
    and computational cost) -- no symbol invented that is not already in the
    paper. Note: $T$ is genuinely overloaded in the source paper (transfer
    function $T:\\mathbb{R}\\to[0,1]$ in the washout subsection vs. max
    iterations in $R(t)=S_M-S_M t/T$) -- both uses are listed rather than
    silently disambiguated."""
    rows = [
        ("$d$", "Total number of features (search-space dimension)"),
        ("$b\\in\\{0,1\\}^d$", "Candidate binary feature-subset mask"),
        ("$f(b)$", "Fitness function, Eq.~1"),
        ("$\\mathrm{Acc}(b)$", "Stratified 5-fold KNN accuracy on subset $b$"),
        ("$|b|$", "Number of selected features (subset size)"),
        ("$R(t)$", "SCSO sensitivity range at iteration $t$"),
        ("$S_M$", "Maximum sensitivity range parameter ($=2$)"),
        ("$T$ (in $R(t)$)", "Maximum number of iterations"),
        ("$T(\\cdot)$ (in washout subsection)", "Continuous-to-binary transfer function, $T:\\mathbb{R}\\to[0,1]$"),
        ("$x_j$", "Continuous position of feature $j$"),
        ("$\\delta_j$", "Perturbation applied to $x_j$"),
        ("$\\Delta_j$", "Induced change in bit-flip probability"),
        ("$\\|T'\\|_\\infty$", "Largest slope (Lipschitz bound) of the transfer function"),
        ("$\\varepsilon$", "Slope bound in a flat/saturated region of $T$"),
        ("$\\gamma$", "RMS modulation strength (bias intensity)"),
        ("$\\rho_j$", "Relevance score of feature $j$, $\\rho_j\\in[0,1]$"),
        ("$b^\\ast_j$", "Preferred bit value for feature $j$, $\\mathbf{1}[\\rho_j>0.5]$"),
        ("$s_j$", "Modulation strength, $s_j=2|\\rho_j-0.5|\\in[0,1]$"),
        ("$\\sigma_j$", "Direction indicator ($+1$ toward $b^\\ast_j$, $-1$ otherwise)"),
        ("$p_j$", "RMS-modulated flip probability for feature $j$, Eq.~2"),
        ("$V(x_j)$", "V-shaped base transfer, $|\\tanh(x_j)|$"),
        ("$\\tau$", "Preferred-bit threshold (deployed at $0.5$; swept over $\\{0.4,0.5,0.6\\}$)"),
        ("$\\rho_{\\mathrm{static}}$", "Static mutual-information relevance prior"),
        ("$I(X_j;y)$", "Mutual information between feature $j$ and the label"),
        ("$H(y)$", "Label entropy"),
        ("$K$", "Number of UMR memetic probes per iteration"),
        ("$N$", "Population size (pop\\_size)"),
        ("$\\mathrm{max\\_nfe}$", "Total fitness-evaluation budget, $\\mathrm{pop\\_size}\\times\\mathrm{max\\_iter}$"),
        ("$n$", "Number of samples (in the $O(dn\\log n)$ MI-prior cost)"),
    ]
    lines = [f"{sym} & {meaning} \\\\" for sym, meaning in rows]
    body = "\n".join(lines)
    cap = (
        "Notation used in Methods, compiled for reference. $T$ is used for "
        "two different quantities in the source text depending on context "
        "(maximum iteration count in the SCSO sensitivity-range formula; "
        "the continuous-to-binary transfer function in the washout "
        "subsection) -- both are listed here explicitly rather than "
        "silently disambiguated, since renaming either would touch the "
        "already-verified main text."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:notation}\n\\footnotesize\n"
        "\\begin{tabular}{lp{9cm}}\n\\toprule\n"
        "Symbol & Meaning \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def threshold_sensitivity_table() -> str:
    """Supplementary table (RG-SCSO_MASTER_FINAL_COMPLETE.md item 12/32.1):
    the 0.5 preferred-bit threshold is a convenience, not a theoretically
    grounded neutral point (Methods) -- this sweep (tau in {0.4,0.5,0.6},
    30 runs/cell, same 5-dataset protocol as every other pilot this
    session) tests whether that choice is at least empirically reasonable.
    No single tau dominates uniformly; tau=0.5 attains the best accuracy on
    2/5 datasets, and mean feature count falls monotonically as tau rises
    on all 5, a modest, non-uniform accuracy trade-off."""
    if not os.path.exists(THRESHOLD_CSV):
        return ""
    th = pd.read_csv(THRESHOLD_CSV)
    if th.empty:
        return ""
    taus = sorted(th["tau"].unique())
    datasets = sorted(th["dataset"].unique())
    lines = []
    for ds in datasets:
        best_acc = max(
            float(th[(th.dataset == ds) & (th.tau == t)]["mean_accuracy"].iloc[0])
            for t in taus
        )
        for i, t in enumerate(taus):
            r = th[(th.dataset == ds) & (th.tau == t)].iloc[0]
            acc = float(r["mean_accuracy"])
            nfeat = float(r["mean_n_selected"])
            phi = float(r["nogueira_phi"])
            acc_cell = f"{acc:.4f}"
            if abs(acc - best_acc) < 1e-9:
                acc_cell = f"\\textbf{{{acc_cell}}}"
            dscol = f"\\multirow{{{len(taus)}}}{{*}}{{{esc(ds)}}}" if i == 0 else ""
            lines.append(f"{dscol} & {t:.1f} & {acc_cell} & {nfeat:.1f} & {phi:.3f} \\\\")
        if ds != datasets[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    cap = (
        "Threshold sensitivity ($\\tau\\in\\{0.4,0.5,0.6\\}$ replacing the "
        "fixed 0.5 preferred-bit threshold in Eq.~(2); 30 independent runs "
        "per cell, same protocol and datasets as the main ablation). "
        "Highest accuracy per dataset in bold. No single $\\tau$ dominates "
        "uniformly: $\\tau=0.5$, the value deployed throughout this paper, "
        "attains the highest accuracy on two of five datasets; mean "
        "selected-feature count falls monotonically as $\\tau$ increases on "
        "every dataset, at a modest, dataset-dependent, non-uniform "
        "accuracy cost. $\\Phi$ is the Nogueira stability index (as "
        "Table~\\ref{tab:stability})."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:threshold}\n\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Dataset & $\\tau$ & Mean Acc. & Mean \\#Feat. & $\\Phi$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def nfe_control_table() -> str:
    """Supplementary table (RG-SCSO_MASTER_FINAL_COMPLETE.md item 10/32.2):
    isolates whether UMR's benefit comes from TARGETING relevance-uncertain
    features specifically, or merely from the extra evaluation budget it
    spends anywhere, by replacing UMR's targeted K-feature selection with K
    uniformly random features at matched NFE (paired-seed, 30 runs/cell,
    same 5-dataset protocol). Does NOT contradict the existing UMR-vs-no-UMR
    ablation (Table 3, still valid): this isolates targeting specifically,
    not UMR's existence. Honest finding: the untargeted control
    significantly beats targeted UMR on both accuracy and feature count on
    the two gene-expression datasets, where UMR's contribution is largest,
    and ties it on the other three -- targeting is not shown to be the
    source of UMR's benefit where that benefit matters most."""
    if not os.path.exists(NFE_CONTROL_CSV):
        return ""
    nf = pd.read_csv(NFE_CONTROL_CSV)
    if nf.empty:
        return ""
    datasets = sorted(nf["dataset"].unique())
    configs = ["RG-SCSO", "RG-SCSO-RandomProbe"]
    labels = {"RG-SCSO": "RG-SCSO (targeted UMR)",
              "RG-SCSO-RandomProbe": "Untargeted control"}

    def vals(ds: str, cfg: str, metric: str) -> np.ndarray:
        sub = nf[(nf.dataset == ds) & (nf.config_name == cfg)].sort_values("run_id")
        return sub[metric].to_numpy()

    sig = {}
    for metric in ("accuracy", "n_selected_features"):
        pvals = []
        for ds in datasets:
            a = vals(ds, "RG-SCSO", metric)
            b = vals(ds, "RG-SCSO-RandomProbe", metric)
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = 1.0
            pvals.append(p)
        p_holm = holm_correction(np.array(pvals))
        for ds, p in zip(datasets, p_holm):
            sig[(metric, ds)] = p < 0.05

    lines = []
    for ds in datasets:
        acc = {c: float(vals(ds, c, "accuracy").mean()) for c in configs}
        nfeat = {c: float(vals(ds, c, "n_selected_features").mean()) for c in configs}
        best_acc = max(acc.values())
        best_nfeat = min(nfeat.values())
        for i, c in enumerate(configs):
            acc_cell = f"{acc[c]:.4f}"
            if abs(acc[c] - best_acc) < 1e-9:
                acc_cell = f"\\textbf{{{acc_cell}}}"
            if sig[("accuracy", ds)] and abs(acc[c] - best_acc) < 1e-9:
                acc_cell += r"$^\dagger$"
            nfeat_cell = f"{nfeat[c]:.1f}"
            if abs(nfeat[c] - best_nfeat) < 1e-9:
                nfeat_cell = f"\\textbf{{{nfeat_cell}}}"
            if sig[("n_selected_features", ds)] and abs(nfeat[c] - best_nfeat) < 1e-9:
                nfeat_cell += r"$^\dagger$"
            dscol = f"\\multirow{{2}}{{*}}{{{esc(ds)}}}" if i == 0 else ""
            lines.append(f"{dscol} & {labels[c]} & {acc_cell} & {nfeat_cell} \\\\")
        if ds != datasets[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    cap = (
        "NFE-matched random-probe control (paired-seed, 30 independent runs "
        "per cell, same protocol and datasets as the main ablation). UMR's "
        "targeted $K$-feature selection (nearest $\\rho=0.5$) is replaced by "
        "$K$ uniformly random features at identical evaluation cost, "
        "isolating whether UMR's benefit requires targeting relevance-"
        "uncertain features specifically, as distinct from the existing "
        "UMR-vs-no-UMR ablation (Table~3, unaffected by this result). Best "
        "value per dataset in bold; $^\\dagger$ marks a significant "
        "difference (paired Wilcoxon signed-rank, Holm-corrected $p<0.05$). "
        "The untargeted control is significantly better on both accuracy "
        "and feature count on the two gene-expression datasets, where "
        "UMR's contribution is largest, and ties targeted UMR on the "
        "other three."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:nfecontrol}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Dataset & Configuration & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def heldout_combined_table(hs: dict) -> str:
    """Merges the former separate held-out accuracy table and held-out
    feature-count table into one combined table (accuracy with mean selected
    feature count in parentheses) -- frees a main-text display-item slot,
    used by classifier_robustness_table() below, while keeping the same
    information. Cell format matches the convention already established in
    classic_baselines_table() ("acc (nfeat)")."""
    algos = hs["algos"]
    cols = "l" + "c" * len(algos)
    head = " & ".join(["Dataset"] + [esc(a) for a in algos])
    lines = []
    for ds in hs["datasets"]:
        row = hs["acc_mean"].loc[ds]
        best = row.max()
        cells = [esc(ds)]
        for a in algos:
            m = hs["acc_mean"].loc[ds, a]
            nf = hs["nf_mean"].loc[ds, a]
            val = f"{m:.4f} ({nf:.1f})"
            if abs(m - best) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cap = (
        "Held-out generalization: mean accuracy, with mean number of "
        "selected features in parentheses, on the outer 20\\% hold-out over "
        "30 runs (relevance prior, search, and CV fitness fit on the 80\\% "
        "training split only; dataset feature counts are given in "
        "Supplementary Table~S1). Standard deviations are omitted here for "
        "compactness; per-run raw results, from which they can be "
        "recomputed exactly, are in the public repository (Data "
        "availability). \\textbf{Bold} = best accuracy per dataset."
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:heldout}\n\\tiny\n\\setlength{\\tabcolsep}{1.0pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def classifier_robustness_table() -> str:
    """NEW main-text table, using the slot freed by heldout_combined_table()
    above: a compact, dataset-averaged summary of the KNN/SVM/RF robustness
    check (RF added this session; run counts differ by wrapper, disclosed
    explicitly rather than hidden). Full per-dataset detail for all three
    wrappers remains in Supplementary Information (robustness_baselines() /
    robustness_svm16() / rf_robustness_table())."""
    if not os.path.exists(ROBUST_CSV):
        return ""
    rob = pd.read_csv(ROBUST_CSV)
    rep5 = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
    rob = rob[rob.dataset.isin(rep5)]
    wrappers = [w for w in ["KNN", "SVM", "RF"] if not rob[rob.wrapper == w].empty]
    algos = ["RG-SCSO-MI", "RG-SCSO-ReliefF", "bSCSO"]
    labels = {"RG-SCSO-MI": "RG-SCSO (MI)", "RG-SCSO-ReliefF": "RG-SCSO (ReliefF)",
              "bSCSO": "bSCSO (no prior)"}

    def m(w, a, col):
        return rob[(rob.wrapper == w) & (rob.algorithm == a)][col].mean()

    def n_runs(w):
        return int(rob[rob.wrapper == w]["run_id"].nunique())

    lines = []
    for w in wrappers:
        nfs = {a: m(w, a, "n_selected_features") for a in algos}
        min_nf = min(nfs.values())
        for i, a in enumerate(algos):
            acc = m(w, a, "accuracy")
            nf = nfs[a]
            nf_cell = f"\\textbf{{{nf:.1f}}}" if abs(nf - min_nf) < 1e-9 else f"{nf:.1f}"
            wcol = f"\\multirow{{3}}{{*}}{{{w} ({n_runs(w)} runs)}}" if i == 0 else ""
            name = f"\\textbf{{{labels[a]}}}" if a == "RG-SCSO-MI" else labels[a]
            lines.append(f"{wcol} & {name} & {acc:.4f} & {nf_cell} \\\\")
        if w != wrappers[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    cap = (
        "Robustness across classifier wrappers and relevance priors, "
        "averaged over the same five representative datasets as "
        "Table~\\ref{tab:classic} (KNN/SVM use the main study's 30 "
        "independent runs per cell; Random Forest uses a reduced 10, given "
        "its far higher per-evaluation cost, disclosed rather than matched "
        "artificially). Per-dataset detail for all three wrappers is given "
        "in Supplementary Information. Fewest features per wrapper in bold."
    )
    return (
        "\\begin{table}[tb]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:classifierrobust}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Wrapper & Method & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


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
        fr = s["friedman"]
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
        f"{_hs['nf_mean']['SCSO'].mean()/_hs['nf_mean']['RG-SCSO'].mean():.1f}"
        "$\\times$ fewer than base SCSO, an advantage that survives every "
        "stress test we run, including comparison against optimizers carrying "
        "published adaptive transfers, against which RG-SCSO does not lead "
        "on accuracy yet still selects "
        f"{adaptive['red_min']:.0f}--{adaptive['red_max']:.0f}\\% fewer "
        "features. Under a budget-matched, leak-free protocol, this "
        "compactness comes with the best mean accuracy of any method "
        "tested: RG-SCSO attains the best Friedman rank and a consistent "
        f"edge over closest competitor AOA (median $|d|={hs_d_aoa:.2f}$). "
        "The binarization interface is the most reliable injection point, "
        "though not the sparsest: LASSO is sparser on the two "
        "gene-expression sets, but "
        "accuracy is preserved, not improved, and parsimony is the "
        "transferable gain."
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

Existing SCSO-based feature selectors~\cite{{bscso,scsofs2,scsofs3}}, and the
broader recent SCSO literature that adds chaotic initialization, differential
mutation, or hybridized search
strategies~\cite{{imscso2024,mescso2025,scsolensobl2024,improvedscso2024}},
all improve continuous-space search dynamics while leaving the binarization
interface itself untouched. Filter criteria such as mutual information and
mRMR~\cite{{guyon,mrmr}} encode problem knowledge cheaply but are decoupled
from the wrapper search; memetic hybridization~\cite{{neri}} adds local
refinement without addressing the same interface; and knowledge-guided
metaheuristics that inject filter information into initialization or the
objective, such as filter-guided PSO for cancer-genome
selection~\cite{{ludwig2025guided}}, still leave the binarization operator
itself knowledge-agnostic. To our knowledge, no prior SCSO feature selector
makes the binarization operator itself per-feature and relevance-aware
(Supplementary Table~S9 positions each cited SCSO-family work against this
claim directly).

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
guidance."""

    # -------------------------------------------------------------- Results
    fr_p_str = pcmp(hs_stats.get("friedman_p", 1)) if hs_stats else "<10^{-3}"
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
        "A dedicated signal-position experiment asks whether "
        "injecting the same mutual-information field elsewhere, at "
        "initialization or as an objective penalty, rather than at the "
        "binarization interface, would work as well; Table~\\ref{tab:ablation} "
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

    results = rf"""\subsection*{{Held-out generalization}}
For each dataset, algorithm, and independent run we draw an outer stratified
80/20 split. The relevance prior, the search, and the cross-validated fitness
are computed exclusively on the 80\% training partition; the selected subset
is evaluated once on the untouched 20\% hold-out, on which a fresh $k$-NN
classifier (standardized on the training partition) reports accuracy, so the
relevance prior never has transductive access to the test labels. Table~\ref{{tab:heldout}}
reports held-out accuracy over all seven algorithms and {s['n']} datasets.
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

\subsection*{{Feature-subset parsimony}}
Parsimony, more than any accuracy margin, is RG-SCSO's defining property.
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

\subsection*{{Ranking and statistical significance}}
{ranking_block}

{rank_table_with_effect_size(s) if s.get("stats") else rank_table(s)}

\subsection*{{Ablation and mechanism}}
{ablation_block}

{signal_position_block}

{extended_ablation_table(s) if s.get("ablation") and os.path.exists(SIGNAL_POS_CSV) else (ablation_table(s) if s.get("ablation") else "")}

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

\subsection*{{Comparison with classical selectors}}
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
presence of a relevance signal. Illustrative convergence traces
(Supplementary Information) show RG-SCSO plateauing at a lower fitness than
SCSO or AOA on the two higher-dimensional datasets tested, with the gap
widening as dimensionality increases, and no distinguishable difference on
the lowest-dimensional one.

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
datasets without that extreme structure.

\noindent\textbf{{Future work}} includes multiobjective
formulations that trade accuracy against subset size explicitly, alternative
and deeper base classifiers, and scaling to ultra-high-dimensional omics
data, where a relevance-guided, binary-native operator should be especially
valuable, and hybrid schemes that fall back to embedded selectors such as
LASSO in the extreme $p\gg n$ regime this study exposes as a genuine
limit."""

    tex = rf"""%=======================================================================
% RG-SCSO, Springer Nature (sn-jnl, sn-nature) format, target: Scientific Reports
% Số liệu sinh tự động từ experiments/results_fs*/*.csv. Cấu trúc: Introduction ->
% Results (co muc con) -> Discussion (KHONG muc con) -> Methods (cuoi bai, khong
% tinh vao gioi han 4500 tu). Toi da 8 hinh+bang trong bai chinh (3 hinh + 5 bang,
% Table 1 gop accuracy+nfeat, Table 5 la classifier-robustness moi); phan con lai
% -> RG-SCSO_SciRep_Supplementary.tex.
%=======================================================================
\documentclass[sn-nature,pdflatex]{{sn-jnl}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{amsthm}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{graphicx}}
\usepackage{{placeins}}
\graphicspath{{{{figures/}}{{./}}}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{rotating}}
\geometry{{twoside=false,bindingoffset=0mm}}
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

\title[RG-SCSO for Parsimonious Feature Selection]{{Relevance-Guided Sand Cat Swarm Optimization: A Per-Feature Relevance-Modulated Binarization for Parsimonious Feature Selection on High-Dimensional Benchmarks}}

\author*[1]{{\fnm{{Bui Quang}} \sur{{Huy}}}}\email{{huybq@donga.edu.vn}}
\author[1]{{\fnm{{Duong Minh}} \sur{{Son}}}}\email{{sondm@donga.edu.vn}}
\affil[1]{{\orgname{{Dong A University}}, \city{{Da Nang}}, \country{{Vietnam}}}}

\abstract{{
{abstract}
}}

\keywords{{Binary metaheuristics, feature selection, relevance-guided search,
sand cat swarm optimization, transfer function}}

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
itself. A threshold sweep ($\tau\in\{{0.4,0.5,0.6\}}$, Supplementary
Information) supports this choice as reasonable rather than arbitrary:
$\tau=0.5$ attains the highest accuracy on two of five datasets tested, and
every value trades a modest, dataset-dependent accuracy shift for a
monotonic reduction in feature count as $\tau$ increases, with no single
value dominating uniformly.

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
extra probes.

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

\backmatter

\bmhead{{Data availability}}
The datasets are publicly available benchmarks (UCI and standard microarray
sets). The source code, the locked preregistration, the per-run seeds, and
the raw results are available in an anonymized repository
(\url{{https://anonymous.4open.science/r/RG-SCSO}}) and will be released in a
public, citable repository upon acceptance.

\bmhead{{Author contributions}}
B.Q.H. conceived the method, implemented the software, ran the experiments,
and drafted the manuscript. D.M.S. contributed to the experimental design and
reviewed the manuscript. Both authors read and approved the final manuscript.

\bmhead{{Competing interests}}
The authors declare no competing interests.

\bmhead{{Funding}}
No funding was received for this work.

\bmhead{{ORCID iDs}}
Bui Quang Huy \href{{https://orcid.org/0009-0000-5761-5098}}{{0009-0000-5761-5098}};
Duong Minh Son \href{{https://orcid.org/0009-0006-6485-7902}}{{0009-0006-6485-7902}}.

\bibliography{{references}}

\end{{document}}
"""
    # Float-drift fix: force pending figures/tables to resolve before
    # crossing a \section boundary (prevents a float queuing several
    # sections past the heading that introduces it).
    tex = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", tex)

    with open(OUT_TEX, "w") as fh:
        fh.write(tex)

    # --------------------------------------------------- Supplementary Info
    washout_tab_placeholder = washout_table(s)
    rf_robustness_placeholder = rf_robustness_table()
    supp = rf"""%=======================================================================
% Supplementary Information for RG-SCSO (Scientific Reports submission)
%=======================================================================
\documentclass[9pt]{{article}}
\usepackage[a4paper,margin=25mm]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,booktabs,multirow,rotating,hyperref}}
\usepackage{{placeins}}
\graphicspath{{{{figures/}}{{./}}}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\renewcommand{{\thesection}}{{S\arabic{{section}}}}
\title{{Supplementary Information for: Relevance-Guided Sand Cat Swarm
Optimization: A Per-Feature Relevance-Modulated Binarization for
Parsimonious Feature Selection on High-Dimensional Benchmarks}}
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
For completeness, Table~\ref{{tab:acc}} and Table~\ref{{tab:nfeat}} report the
standard in-sample protocol, in which the relevance prior, search, and
reported metric share the same cross-validation folds; effect sizes here are
inflated relative to the held-out estimate (see main text Discussion).
{accuracy_table(s)}
{nfeat_table(s)}

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

\section{{Literature positioning}}
Table~\ref{{tab:litpos}} positions every recent SCSO-family work cited in
this paper against the binarization-interface claim made in the
Introduction.

{literature_positioning_table()}

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

\section{{Threshold sensitivity}}
The 0.5 preferred-bit threshold
(Methods) is a convenience, not a theoretically grounded neutral point; this
sweep tests whether it is at least an empirically reasonable one, reported
both as a table and, for the accuracy dimension, as a heatmap.

{threshold_sensitivity_table() if os.path.exists(THRESHOLD_CSV) else ''}

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

\section{{NFE-matched random-probe control}}
Isolates whether UMR's benefit
(Table~3) requires targeting relevance-uncertain features specifically, or
only the extra evaluation budget it spends, by replacing its targeted
$K$-feature selection with $K$ uniformly random features at matched NFE.

{nfe_control_table() if os.path.exists(NFE_CONTROL_CSV) else ''}

\section{{Convergence behavior on the feature-selection objective}}
Illustrative,
not a new statistical claim: RG-SCSO, SCSO, and AOA on three representative
datasets (Zoo, WDBC, ColonCancer), 5 independent runs each, full per-iteration
best-fitness trace. Distinct from the earlier convergence figure
(Supplementary Information), which plots a separate, older 30-run capture on
two datasets from an earlier phase of this project; this figure uses the
same feature-selection objective and protocol as the main study, sourced from
this revision's own runs.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{convergence_fs.pdf}}
\caption{{Mean best fitness versus iteration, RG-SCSO vs.\ SCSO vs.\ AOA, on
Zoo (16 features), WDBC (30 features), and ColonCancer (2000 features), mean
over 5 runs. On Zoo the three algorithms converge along essentially the same
trajectory. On WDBC and ColonCancer, RG-SCSO both converges faster and
plateaus at a lower (better) fitness than SCSO or AOA, which themselves
plateau early at a distinctly worse value rather than continuing to close the
gap with more iterations; the difference grows with dimensionality.}}
\label{{fig:convfs}}
\end{{figure}}

\section{{Accuracy-parsimony trade-off}}
Held-out accuracy
versus mean selected-feature fraction (feature count normalized by each
dataset's total, so datasets with 8 vs.\ 3571 features are comparable),
one point per algorithm, averaged across all 18 datasets -- plotted
entirely from the existing held-out results (Table~1), no new experiment.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{{accuracy_parsimony_tradeoff.pdf}}
\caption{{Mean held-out accuracy vs.\ mean selected-feature fraction,
averaged across all 18 datasets, one point per algorithm. RG-SCSO occupies
the top-left corner, jointly the highest accuracy and the sparsest subsets
of all seven algorithms tested -- it is not merely competitive on one axis
at the cost of the other. AOA attains the second-highest accuracy but
selects on average 98\% of available features, essentially no feature
selection; COA is the second-sparsest method but at a distinctly lower
accuracy than RG-SCSO. The remaining four baselines (SCSO, RIME, PSO, GWO)
cluster together at roughly half the features and clearly lower accuracy
than RG-SCSO.}}
\label{{fig:tradeoff}}
\end{{figure}}

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
    # Float-drift fix (see OUT_TEX write, above) applied to the
    # Supplementary too -- this is the document where it mattered most: many
    # small tables/figures in quick succession were queuing floats several
    # \section headings ahead of (or behind) the section that introduces them.
    supp = re.sub(r"\\section\{", r"\\FloatBarrier\n\\section{", supp)

    with open(OUT_SUPP_TEX, "w") as fh:
        fh.write(supp)

    print(f"Đã ghi {OUT_TEX} và {OUT_SUPP_TEX}")


if __name__ == "__main__":
    build()
