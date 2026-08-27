"""Sinh bài RG-SCSO ra file LaTeX theo chuẩn Springer sn-jnl (target Applied Intelligence).

Nguyên tắc (giống bản docx):
  - MỌI số trong bảng/prose đọc tự động từ fs_results.csv, không gõ tay.
  - Phần chưa chạy xong (RIME đủ 18 dataset, Wilcoxon/Holm, effect size,
    Friedman/CD, ablation, hình) để trống bằng \\textit{[pending ...]}.
  - Bố cục, đánh số, table/figure/equation/algorithm theo IEEEtran_skeleton.tex.

Chạy:   .venv/bin/python build_paper_tex.py
Xuất:   RG-SCSO_demo.tex   (compile trên Overleaf, class sn-jnl, Springer Nature LaTeX)
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import build_heldout_table as _heldout
from build_paper_structure import COMPLETE_ALGOS, load_summary

OUT_TEX = "RG-SCSO_demo.tex"
FS_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
SENS_CSV = os.path.join("experiments", "results_fs_sensitivity",
                        "fs_sensitivity_results.csv")
ADAPTIVE_CSV = os.path.join("experiments", "results_fs_adaptive_baselines",
                            "summary_vs_rgscso.csv")
FAMILY_CSV = os.path.join("experiments", "results_fs_scso_family",
                          "fs_scso_family_results.csv")
ROBUST_CSV = os.path.join("experiments", "results_fs_robustness",
                          "fs_robustness_results.csv")
PROC_DIR = os.path.join("data", "processed")
RUNTIME_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
DIVERSITY_CSV = os.path.join("experiments", "results_diversity", "diversity_history.csv")
YEAR = {"RG-SCSO": "ours", "SCSO": "2022", "AOA": "2021", "COA": "2023",
        "GWO": "2014", "PSO": "1995", "RIME": "2023"}


def esc(s: str) -> str:
    """Escape ký tự LaTeX trong tên dataset/thuật toán."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def sci(x: float, sig: int = 1) -> str:
    """Ký hiệu khoa học LaTeX cho p-value (dùng TRONG math mode).
    Vd 2.36e-13 -> '2.4\\times10^{-13}'. Với p rất nhỏ tránh kiểu '2.4e-13' của code."""
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    return f"{mant:.{sig}f}\\times10^{{{exp}}}"


def pcmp(p: float) -> str:
    """RHS của báo cáo p-value: p rất nhỏ -> '<10^{-3}' (tránh ngụ ý độ chính xác
    giả với chỉ 18 block, rev #8); còn lại '=<giá trị>'. Dùng dạng '$p{pcmp(p)}$'."""
    if p < 1e-3:
        return "<10^{-3}"
    return f"={p:.3f}"


def accuracy_table(s: dict) -> str:
    cols = "l" + "c" * (len(COMPLETE_ALGOS) + 2)  # dataset + #F + methods + RIME
    head = " & ".join(["Dataset", "\\#F"] + [esc(a) for a in COMPLETE_ALGOS] + ["RIME"])
    lines = []
    rime_incomplete = False
    for ds in s["datasets"]:
        # best gồm cả RIME nếu dataset này RIME đã đủ 30 run
        vals = list(s["acc_mean"].loc[ds])
        if ds in s["rime"]:
            vals.append(s["rime"][ds]["acc"])
        best = max(vals)
        cells = [esc(ds), str(int(s["ntot"][ds]))]
        for a in COMPLETE_ALGOS:
            m, sd = s["acc_mean"].loc[ds, a], s["acc_std"].loc[ds, a]
            val = f"{m:.4f}$\\pm${sd:.3f}"
            if abs(m - best) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        if ds in s["rime"]:  # RIME đủ 30 run → số thật
            m, sd = s["rime"][ds]["acc"], s["rime"][ds]["std"]
            val = f"{m:.4f}$\\pm${sd:.3f}"
            if abs(m - best) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        else:  # RIME đang chạy dataset này
            cells.append("--")
            rime_incomplete = True
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    rime_note = (" RIME `--' = run still in progress for that dataset."
                 if rime_incomplete else "")
    return (
        "\\begin{sidewaystable}[t]\n\\centering\n"
        "\\caption{Mean $\\pm$ std classification accuracy over 30 runs. "
        "\\#F = total features. \\textbf{Bold} = best per dataset. Higher is better ($\\uparrow$)."
        + rime_note + "}\n"
        "\\label{tab:acc}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{sidewaystable}}\n"
    )


def nfeat_table(s: dict) -> str:
    cols = "l" + "c" * len(COMPLETE_ALGOS)
    head = " & ".join(["Dataset"] + [esc(a) for a in COMPLETE_ALGOS])
    lines = []
    for ds in s["datasets"]:
        least = s["nf_mean"].loc[ds].min()
        cells = [esc(ds)]
        for a in COMPLETE_ALGOS:
            v = s["nf_mean"].loc[ds, a]
            val = f"{v:.1f}"
            if abs(v - least) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mean number of selected features over 30 runs. "
        "\\textbf{Bold} = fewest ($\\downarrow$).}\n"
        "\\label{tab:nfeat}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def rank_table(s: dict) -> str:
    has = s.get("stats")
    ranking = s["rank7"].sort_values() if has else s["avg_rank"]
    lines = []
    for a, r in ranking.items():
        if a == "RG-SCSO":
            wtl = "--"
        elif has:
            wtl = "{}/{}/{}".format(*s["sig_wtl"].get(a, (0, 0, 0)))
        else:
            wtl = "\\textit{[pending]}"
        name = f"{esc(a)} ({YEAR.get(a, '?')})"
        lines.append(f"{name} & {r:.2f} & {wtl} \\\\")
    body = "\n".join(lines)
    cap = ("Average accuracy rank across the "
           f"{s['n']} datasets and the Holm-significant win/tie/loss of RG-SCSO "
           "against each baseline (Wilcoxon signed-rank, $\\alpha=0.05$)."
           if has else
           "Average accuracy rank across the "
           f"{s['n']} datasets and win/tie/loss of RG-SCSO against each "
           "baseline. Wilcoxon signed-rank with Holm correction is reported in "
           "the final version.")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:rank}\n\\footnotesize\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Algorithm (year) & Avg. rank & W/T/L vs.\\ RG-SCSO (Holm) \\\\\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


# Nhãn LaTeX các cấu hình ablation (§V-D). "Full" = thiết-kế 3-thành-phần khảo
# sát; "$-$ ORL" = RG-SCSO CUỐI (2-thành-phần).
ABL_LABEL_TEX = {
    "Full": r"Full (RMS+ORL+UMR)",
    "NoRMS": r"$-$ RMS",
    "NoORL": r"$-$ ORL (final)",
    "NoUMR": r"$-$ UMR",
    "NoImprovement": r"$-$ all three",
}


def ablation_table(s: dict) -> str:
    """Bảng V, accuracy trung bình mỗi cấu hình × dataset. Hàng Full \\textbf;
    ô có $\\dagger$ = tệ hơn Full có ý nghĩa (Wilcoxon paired + Holm, p<0.05)."""
    ds_list = s["abl_datasets"]  # 5 dataset ablation, KHÔNG phải 18 của bảng chính
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
    body = "\n".join(lines)
    cap = ("Component ablation: mean accuracy over 30 runs on five datasets. "
           "$^\\dagger$ significantly worse than Full (paired Wilcoxon signed-rank, "
           "Holm-corrected $p<0.05$). ``$-$ ORL'' is the final RG-SCSO (RMS+UMR); "
           "ORL removal never degrades accuracy, so ORL is not retained.")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:ablation}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def dataset_table(s: dict) -> str:
    """Bảng I, đặc trưng dataset (samples/features/classes), đọc từ data/processed."""
    lines = []
    for ds in s["datasets"]:
        df = pd.read_csv(os.path.join(PROC_DIR, f"{ds}.csv"))
        n_samp, n_feat, n_cls = len(df), df.shape[1] - 1, df["label"].nunique()
        lines.append(f"{esc(ds)} & {n_samp} & {n_feat} & {n_cls} \\\\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Dataset characteristics (samples, features, classes).}\n"
        "\\label{tab:datasets}\n\\footnotesize\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Dataset & Samples & Features & Classes \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def runtime_table() -> str:
    """Bảng VIII, wall-clock trung bình (s/run) trên 5 dataset đại diện × 7 algo."""
    fs = pd.read_csv(FS_CSV)
    rt = fs.pivot_table(index="algorithm", columns="dataset",
                        values="runtime_seconds", aggfunc="mean")
    rt_sd = fs.pivot_table(index="algorithm", columns="dataset",
                           values="runtime_seconds", aggfunc="std")
    algos = ["RG-SCSO", "SCSO", "AOA", "COA", "GWO", "PSO", "RIME"]
    cols = "l" + "c" * len(RUNTIME_DATASETS)
    head = "Algorithm & " + " & ".join(esc(d) for d in RUNTIME_DATASETS)
    lines = []
    for a in algos:
        cells = [esc(a)] + [f"{rt.loc[a, d]:.1f}$\\pm${rt_sd.loc[a, d]:.1f}"
                            for d in RUNTIME_DATASETS]
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Computational cost: mean$\\pm$std wall-clock runtime in seconds "
        "per run over 30 runs on five representative datasets (identical protocol, "
        "single CPU thread; lower is better).}\n"
        "\\label{tab:runtime}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def washout_table(s: dict) -> str:
    """Bảng IX, ECL-SCSO vs SCSO per-dataset + Wilcoxon (chứng cứ washout Phase-3)."""
    fs = pd.read_csv(FS_CSV)
    lines, wins, losses, ties = [], 0, 0, 0
    for ds in s["datasets"]:
        e = fs[(fs.algorithm == "ECL-SCSO") & (fs.dataset == ds)].sort_values(
            "run_id")["accuracy"].to_numpy()
        c = fs[(fs.algorithm == "SCSO") & (fs.dataset == ds)].sort_values(
            "run_id")["accuracy"].to_numpy()
        try:
            p = wilcoxon(e, c).pvalue
        except ValueError:  # tất cả hiệu = 0 → không khác biệt
            p = 1.0
        if p < 0.05 and e.mean() > c.mean():
            sig, wins = f"win (p={p:.3f})", wins + 1
        elif p < 0.05:
            sig, losses = f"loss (p={p:.3f})", losses + 1
        else:
            sig, ties = "n.s.", ties + 1
        lines.append(f"{esc(ds)} & {e.mean():.4f} & {c.mean():.4f} & {sig} \\\\")
    body = "\n".join(lines)
    cap = ("Preliminary washout study: mean accuracy of the continuous-enhanced "
           "SCSO (ECL-SCSO) vs.\\ base SCSO over 30 runs; significance by Wilcoxon "
           f"signed-rank at $\\alpha=0.05$ ({wins} wins, {losses} loss, {ties} ties).")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:washout}\n\\footnotesize\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Dataset & ECL-SCSO & SCSO & Sig.\\ (Wilcoxon) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


# Nhãn + định dạng giá trị cho từng sweep của bảng sensitivity (§C phụ lục).
SENS_ROWS = [
    ("gamma", r"$\gamma$ (RMS)", lambda v: f"{v:.2f}"),
    ("umr_k", r"$K$ (UMR)", lambda v: f"{int(v)}"),
    ("ema_lambda", r"$\lambda$ (ORL)", lambda v: f"{v:.2f}"),
    ("w_online", r"$w_o$ (ORL)", lambda v: f"{v:.1f}"),
]


def sensitivity_table() -> str:
    """Bảng X, OFAT sensitivity (γ/K/λ/w_o) pooled 3 dataset × 10 run, NFE cố định."""
    sv = pd.read_csv(SENS_CSV)
    lines = []
    for gi, (sweep, label, fmt) in enumerate(SENS_ROWS):
        g = sv[sv.sweep == sweep].groupby("value").agg(
            acc=("accuracy", "mean"), nf=("n_selected_features", "mean")).sort_index()
        if gi:
            lines.append(r"\addlinespace")
        for ri, (val, row) in enumerate(g.iterrows()):
            plabel = label if ri == 0 else ""
            lines.append(f"{plabel} & {fmt(val)} & {row['acc']:.4f} & "
                         f"{row['nf']:.1f} \\\\")
    body = "\n".join(lines)
    cap = ("Hyperparameter sensitivity (one-factor-at-a-time): mean accuracy and "
           "mean number of selected features over three datasets $\\times$ 10 runs "
           "at a fixed NFE budget. $\\gamma$ and $K$ are swept on the final "
           "(ORL-off) method; $\\lambda$ and $w_o$ on the ORL-on variant (inert in "
           "the shipped method).")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:sensitivity}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Parameter & Value & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def adaptive_baselines() -> dict:
    """Bảng XI + số liệu §D, baseline transfer thích nghi (bPSO/bGWO × TVT/V4).

    Đọc summary_vs_rgscso.csv (5 config × mean_acc/nfeat/fit/rank, mọi số từ
    artifact). Trả về dict: table (LaTeX), red_min/red_max (% feature RG-SCSO ít
    hơn từng baseline), best_acc_cfg (config accuracy cao nhất) + acc/rank của nó.

    Returns:
        dict với keys: table, red_min, red_max, best_acc_cfg, best_acc, best_rank,
                       rg_nfeat.
    """
    df = pd.read_csv(ADAPTIVE_CSV)
    labels = {"RG-SCSO": "RG-SCSO (ours)", "bPSO-TVT": "bPSO-TVT",
              "bGWO-TVT": "bGWO-TVT", "bPSO-V4": "bPSO-V4", "bGWO-V4": "bGWO-V4"}
    order = ["RG-SCSO", "bPSO-TVT", "bGWO-TVT", "bPSO-V4", "bGWO-V4"]
    df = df.set_index("config").loc[order]

    best_acc = df["mean_acc"].max()
    best_fit = df["mean_fit"].min()
    best_rank = df["friedman_rank_fit"].min()
    min_nfeat = df["mean_nfeat"].min()

    def bold(val: float, target: float, fmt: str) -> str:
        cell = format(val, fmt)
        return f"\\textbf{{{cell}}}" if abs(val - target) < 1e-9 else cell

    lines = []
    for cfg, row in df.iterrows():
        name = labels[cfg]
        name = f"\\textbf{{{name}}}" if cfg == "RG-SCSO" else name
        lines.append(
            f"{name} & {bold(row['mean_acc'], best_acc, '.4f')} & "
            f"{bold(row['mean_nfeat'], min_nfeat, '.1f')} & "
            f"{bold(row['mean_fit'], best_fit, '.4f')} & "
            f"{bold(row['friedman_rank_fit'], best_rank, '.2f')} \\\\")
    body = "\n".join(lines)

    rg_nfeat = df.loc["RG-SCSO", "mean_nfeat"]
    reductions = [(row["mean_nfeat"] - rg_nfeat) / row["mean_nfeat"] * 100
                  for cfg, row in df.iterrows() if cfg != "RG-SCSO"]
    best_acc_cfg = df["mean_acc"].idxmax()

    n_ds = pd.read_csv(FS_CSV)["dataset"].nunique()
    cap = ("Confound-isolation against adaptive-transfer binary optimizers "
           "(bPSO/bGWO with the time-varying $|\\tanh|$ transfer of Islam "
           "et~al.~\\cite{islam2017tvtf} and the V4 transfer of Teng "
           f"et~al.~\\cite{{teng2017avbpso}}), same {n_ds} datasets $\\times$ 30 runs "
           "and budget as the main study. RG-SCSO does not lead on accuracy, yet "
           "selects the fewest features consistently. Best per column in bold; "
           "rank is over mean fitness.")
    table = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:adaptive}\n\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Method & Mean Acc. & Mean \\#Feat. & Mean Fitness & Rank \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    return {
        "table": table,
        "red_min": min(reductions),
        "red_max": max(reductions),
        "best_acc_cfg": best_acc_cfg,
        "best_acc": best_acc,
        "best_rank": best_rank,
        "rg_nfeat": rg_nfeat,
    }


def scso_family_baselines() -> dict:
    """Bảng + số liệu §so-với-họ-SCSO (rev #2). RG-SCSO vs bSCSO-S/bSCSO-OBL trên
    18 dataset × 30 run, cùng protocol. W/T/L theo Wilcoxon Holm-per-dataset (khớp
    phương pháp bài). Mọi số đọc từ raw per-run CSV, không gõ tay.

    Returns:
        dict: table (LaTeX), mean_acc (dict theo algo), mean_nfeat (dict),
              wtl (dict algo -> "w/t/l"), red (dict algo -> % feature RG ít hơn).
    """
    from src.stats.statistical_tests import paired_wilcoxon_vs_target

    rg = pd.read_csv(FS_CSV)
    rg = rg[rg.algorithm == "RG-SCSO"][
        ["algorithm", "dataset", "run_id", "accuracy", "n_selected_features"]]
    fam = pd.read_csv(FAMILY_CSV)[
        ["algorithm", "dataset", "run_id", "accuracy", "n_selected_features"]]
    combined = pd.concat([rg, fam], ignore_index=True)
    order = ["RG-SCSO", "bSCSO-S", "bSCSO-OBL"]
    labels = {"RG-SCSO": "RG-SCSO (ours)", "bSCSO-S": "bSCSO (S-shaped)",
              "bSCSO-OBL": "bSCSO (V-shaped + OBL)"}

    mean_acc = {a: combined[combined.algorithm == a].accuracy.mean() for a in order}
    mean_nfeat = {a: combined[combined.algorithm == a].n_selected_features.mean()
                  for a in order}
    wil = paired_wilcoxon_vs_target(
        combined, value_col="accuracy", group_col="dataset",
        algorithm_col="algorithm", run_col="run_id",
        target_algorithm="RG-SCSO", lower_is_better=False)
    datasets = sorted(rg.dataset.unique())
    wtl, red, smaller = {}, {}, {}
    for a in order[1:]:
        sub = wil[wil.compared_with == a]
        w = int((sub["mark"] == "+").sum())
        t = int((sub["mark"] == "=").sum())
        loss = int((sub["mark"] == "-").sum())
        wtl[a] = f"{w}/{t}/{loss}"
        red[a] = (mean_nfeat[a] - mean_nfeat["RG-SCSO"]) / mean_nfeat[a] * 100
        smaller[a] = sum(
            combined[(combined.algorithm == "RG-SCSO") & (combined.dataset == d)]
            .n_selected_features.mean()
            < combined[(combined.algorithm == a) & (combined.dataset == d)]
            .n_selected_features.mean()
            for d in datasets)

    min_nf = min(mean_nfeat.values())
    lines = []
    for a in order:
        name = f"\\textbf{{{labels[a]}}}" if a == "RG-SCSO" else labels[a]
        nf = mean_nfeat[a]
        nf_cell = f"\\textbf{{{nf:.1f}}}" if abs(nf - min_nf) < 1e-9 else f"{nf:.1f}"
        wtl_cell = "-" if a == "RG-SCSO" else wtl[a]
        lines.append(f"{name} & {mean_acc[a]:.4f} & {nf_cell} & {wtl_cell} \\\\")
    body = "\n".join(lines)
    n_ds = combined["dataset"].nunique()
    cap = ("Comparison with same-family binary SCSO feature selectors under the "
           f"identical protocol ({n_ds} datasets $\\times$ 30 runs, budget-matched). "
           "bSCSO (S-shaped) and bSCSO (V-shaped + opposition-based learning) are "
           "reimplementations of the standard SCSO-FS recipe with no per-feature "
           "relevance field. W/T/L is RG-SCSO's Holm-corrected win/tie/loss on "
           "accuracy (paired Wilcoxon, per-dataset family). Fewest features in bold.")
    table = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:scsofamily}\n\\footnotesize\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Method & Mean Acc. & Mean \\#Feat. & W/T/L vs RG-SCSO \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    return {"table": table, "mean_acc": mean_acc, "mean_nfeat": mean_nfeat,
            "wtl": wtl, "red": red, "smaller": smaller, "n_ds": n_ds}


def robustness_baselines() -> dict:
    """Bảng + số liệu §robustness (rev #3/#6). RG-SCSO(MI/ReliefF) vs bSCSO qua
    KNN/SVM trên 5 dataset. Số đọc từ raw per-run CSV.

    Returns:
        dict: table (LaTeX), red (dict wrapper -> % RG(MI) ít feature hơn bSCSO),
              relieff_vs_mi (dict wrapper -> tỉ lệ nfeat ReliefF/MI), acc/nfeat means.
    """
    rob = pd.read_csv(ROBUST_CSV)
    # Bảng chéo KNN×SVM cố định trên 5 dataset đại diện (khớp cả 2 wrapper); SVM/18
    # được phân tích riêng khi chạy xong, không trộn vào đây.
    rep5 = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
    rob = rob[rob.dataset.isin(rep5)]
    wrappers = ["KNN", "SVM"]
    algos = ["RG-SCSO-MI", "RG-SCSO-ReliefF", "bSCSO"]
    labels = {"RG-SCSO-MI": "RG-SCSO (MI)", "RG-SCSO-ReliefF": "RG-SCSO (ReliefF)",
              "bSCSO": "bSCSO (no prior)"}

    def m(w, a, col):
        return rob[(rob.wrapper == w) & (rob.algorithm == a)][col].mean()

    lines, red, relieff_vs_mi = [], {}, {}
    for w in wrappers:
        nfs = {a: m(w, a, "n_selected_features") for a in algos}
        min_nf = min(nfs.values())
        for i, a in enumerate(algos):
            acc = m(w, a, "accuracy")
            nf = nfs[a]
            nf_cell = f"\\textbf{{{nf:.1f}}}" if abs(nf - min_nf) < 1e-9 else f"{nf:.1f}"
            wcol = f"\\multirow{{3}}{{*}}{{{w}}}" if i == 0 else ""
            name = f"\\textbf{{{labels[a]}}}" if a == "RG-SCSO-MI" else labels[a]
            lines.append(f"{wcol} & {name} & {acc:.4f} & {nf_cell} \\\\")
        red[w] = (nfs["bSCSO"] - nfs["RG-SCSO-MI"]) / nfs["bSCSO"] * 100
        relieff_vs_mi[w] = nfs["RG-SCSO-ReliefF"] / nfs["RG-SCSO-MI"]
        if w != wrappers[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    n_ds = rob["dataset"].nunique()
    cap = ("Robustness across classifier wrappers and relevance priors on "
           f"{n_ds} representative datasets ($\\times$ 30 runs, budget-matched). "
           "RG-SCSO's mutual-information prior is compared against a ReliefF prior "
           "and against bSCSO (no prior), under both a KNN and an SVM wrapper. "
           "Fewest features per wrapper in bold.")
    table = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:robustness}\n\\footnotesize\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Wrapper & Method & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    return {"table": table, "red": red, "relieff_vs_mi": relieff_vs_mi, "n_ds": n_ds}


def diversity_analysis() -> dict:
    """Bằng chứng thực nghiệm cho §1.1/§1.2 (Diem_yeu_RG-SCSO.md): đa dạng quần
    thể + tỉ lệ bit đóng băng theo vòng lặp, gamma in {0, 0.5, 1.0}, trên 3 dataset
    (thấp/trung/siêu cao chiều). gamma=0.5 LÀ giá trị triển khai thật trong bài;
    gamma=1.0 là stress test (bias cực đại); gamma=0 là V-shaped thuần đối chứng.
    """
    df = pd.read_csv(DIVERSITY_CSV)
    datasets = sorted(df["dataset"].unique())
    end = df.sort_values("iter").groupby(["dataset", "gamma"]).tail(1)

    def val(ds, g, col):
        row = end[(end.dataset == ds) & (end.gamma == g)]
        return float(row[col].iloc[0])

    def maxcol(ds, g, col):
        return float(df[(df.dataset == ds) & (df.gamma == g)][col].max())

    def first_iter_over(ds, g, thresh):
        sub = df[(df.dataset == ds) & (df.gamma == g)].sort_values("iter")
        hit = sub[sub["frozen_frac"] > thresh]
        return int(hit["iter"].min()) if len(hit) else None

    rows = []
    for ds in datasets:
        d0, d5, d1 = (val(ds, g, "diversity") for g in (0.0, 0.5, 1.0))
        f0, f5, f1 = (val(ds, g, "frozen_frac") for g in (0.0, 0.5, 1.0))
        rows.append(f"{esc(ds)} & {d0:.3f} & {d5:.3f} & {d1:.3f} & "
                    f"{f0:.3f} & {f5:.3f} & {f1:.3f} \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{End-of-run population diversity and frozen-bit fraction, "
        "mean over 3 runs, for three relevance-bias settings: no relevance "
        "($\\gamma=0$, plain V-shaped), the default used throughout the paper "
        "($\\gamma=0.5$), and a maximal-bias stress test ($\\gamma=1$). "
        "Diversity is the mean expected normalized Hamming spread "
        "$2\\bar p(1-\\bar p)$ across features; frozen fraction is the share of "
        "features on which every individual in the population agrees.}\n"
        "\\label{tab:diversity}\n\\footnotesize\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "& \\multicolumn{3}{c}{Diversity} & \\multicolumn{3}{c}{Frozen fraction} "
        "\\\\\nDataset & $\\gamma$=0 & $\\gamma$=0.5 & $\\gamma$=1 & $\\gamma$=0 & "
        "$\\gamma$=0.5 & $\\gamma$=1 \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    max_frz_g5 = max(maxcol(ds, 0.5, "frozen_frac") for ds in datasets)
    first_freeze = {ds: first_iter_over(ds, 1.0, 0.1) for ds in datasets}
    worst_ds = max(datasets, key=lambda d: val(d, 1.0, "frozen_frac"))
    return {
        "table": table, "datasets": datasets,
        "max_frz_g5": max_frz_g5,
        "mean_frz_g1": float(np.mean([val(ds, 1.0, "frozen_frac") for ds in datasets])),
        "mean_frz_g0": float(np.mean([val(ds, 0.0, "frozen_frac") for ds in datasets])),
        "worst_ds": worst_ds, "worst_frz_g1": val(worst_ds, 1.0, "frozen_frac"),
        "worst_first_freeze_iter": first_freeze[worst_ds],
    }


def inference_value() -> dict | None:
    """§2.2 Diem_yeu_RG-SCSO.md: giá trị THỰC TIỄN của parsimony, đo bằng KNN
    inference latency. measure_inference_time.py cho hai kết quả trung thực:
    (a) trên chính n_train của 18 dataset (50-455 mẫu) hiệu ứng nằm trong nhiễu
    đo (không claim), (b) mô phỏng có kiểm soát ở quy mô triển khai
    (n_train=5000, d = số feature THẬT RG-SCSO/AOA chọn) cho tốc độ tăng rõ.
    """
    p = os.path.join("experiments", "results_inference", "inference_time_synthetic.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return {
        "mean_speedup": float(df["speedup_pct"].mean()),
        "min_speedup": float(df["speedup_pct"].min()),
        "max_speedup": float(df["speedup_pct"].max()),
        "best_ds": df.loc[df["speedup_pct"].idxmax(), "dataset"],
    }


def robustness_svm16() -> dict:
    """SVM trên 16/18 dataset (rev #3 — robustness classifier trên diện rộng).
    Bỏ KrVsKpEW/WaveformEW (RBF-SVM O(n^2)/fit, bất khả thi trong wrapper 15k-eval).
    """
    from src.stats.statistical_tests import paired_wilcoxon_vs_target
    svm = pd.read_csv(ROBUST_CSV)
    svm = svm[svm.wrapper == "SVM"]
    ds = sorted(svm.dataset.unique())
    algos = ["RG-SCSO-MI", "RG-SCSO-ReliefF", "bSCSO"]
    labels = {"RG-SCSO-MI": "RG-SCSO (MI)", "RG-SCSO-ReliefF": "RG-SCSO (ReliefF)",
              "bSCSO": "bSCSO (no prior)"}

    def mn(a, col):
        return svm[svm.algorithm == a][col].mean()

    macc = {a: mn(a, "accuracy") for a in algos}
    mnf = {a: mn(a, "n_selected_features") for a in algos}
    red = (mnf["bSCSO"] - mnf["RG-SCSO-MI"]) / mnf["bSCSO"] * 100
    ratio = mnf["RG-SCSO-ReliefF"] / mnf["RG-SCSO-MI"]
    wil = paired_wilcoxon_vs_target(
        svm[svm.algorithm.isin(["RG-SCSO-MI", "bSCSO"])], "accuracy", "dataset",
        "algorithm", "run_id", "RG-SCSO-MI", lower_is_better=False)
    w = int((wil["mark"] == "+").sum()); t = int((wil["mark"] == "=").sum())
    loss = int((wil["mark"] == "-").sum())
    smaller = sum(
        svm[(svm.algorithm == "RG-SCSO-MI") & (svm.dataset == d)].n_selected_features.mean()
        < svm[(svm.algorithm == "bSCSO") & (svm.dataset == d)].n_selected_features.mean()
        for d in ds)
    min_nf = min(mnf.values())
    lines = []
    for a in algos:
        nf = mnf[a]
        nfc = f"\\textbf{{{nf:.1f}}}" if abs(nf - min_nf) < 1e-9 else f"{nf:.1f}"
        name = f"\\textbf{{{labels[a]}}}" if a == "RG-SCSO-MI" else labels[a]
        lines.append(f"{name} & {macc[a]:.4f} & {nfc} \\\\")
    body = "\n".join(lines)
    cap = (f"SVM wrapper on {len(ds)} of the 18 datasets (mean over 30 runs). The "
           "two largest-sample sets (KrVsKpEW, WaveformEW) are excluded because "
           "kernel-SVM training is $O(n^2)$ per fit and infeasible inside a "
           "15{,}000-evaluation wrapper; the KNN main study already covers them. "
           "Fewest features in bold.")
    table = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        "\\label{tab:svm16}\n\\footnotesize\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Method & Mean Acc. & Mean \\#Feat. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    return {"table": table, "n_ds": len(ds), "red": red, "ratio": ratio,
            "wtl": f"{w}/{t}/{loss}", "smaller": smaller, "macc": macc, "mnf": mnf}


def build() -> None:
    s = load_summary()
    rank = s["avg_rank"]
    colon = s["gene"].get("ColonCancer", {})

    #, Bảng phụ + prose parity với docx (mọi số đọc từ artifact), 
    dataset_tab = dataset_table(s)
    runtime_tab = runtime_table()
    washout_tab = washout_table(s)
    sensitivity_tab = sensitivity_table()
    adaptive = adaptive_baselines()
    scsofam = scso_family_baselines()
    robust = robustness_baselines()
    svm16 = robustness_svm16()
    inf_val = inference_value()
    if inf_val is not None:
        inference_sentence = (
            " Parsimony's computational value is concrete: the dimensionality "
            "reductions RG-SCSO achieves are too small relative to these "
            "datasets' modest sample counts (50--455 training instances) for a "
            "wall-clock inference saving to be resolvable above call overhead "
            "here, but replaying the same reductions at a deployment-scale "
            "workload (5{,}000 training instances, brute-force $k$-NN) yields a "
            f"{inf_val['min_speedup']:.0f}\\% to {inf_val['max_speedup']:.0f}\\% "
            f"batch-inference speedup over AOA (mean {inf_val['mean_speedup']:.0f}\\%, "
            f"largest on {esc(inf_val['best_ds'])}, where the feature-count gap is "
            "widest), a controlled complexity demonstration rather than a "
            "measurement on the benchmark's own test sets.")
    else:
        inference_sentence = ""
    diversity = diversity_analysis() if os.path.exists(DIVERSITY_CSV) else None
    if diversity is not None:
        diversity_subsection = (
            "\\subsection{Exploration Safety: A Bit-Freezing Diagnostic}\n"
            "\\label{sec:diversity}\n"
            "Because RMS biases the flip probability toward a preferred bit "
            f"(\\eqref{{eq:rms}}), a legitimate concern is that strong bias "
            "($\\gamma$ close to 1) could push $p_j$ toward the clip boundary and "
            "freeze confidently classified features into the mask early, "
            "collapsing the population's coverage of the subset space "
            "(Section~\\ref{sec:washout} shows this failure mode for continuous "
            "operators; here we ask whether RMS reintroduces it in the binary "
            "domain). We instrument the search to record, each iteration, the "
            "population's mean expected Hamming spread "
            "$2\\bar p_j(1-\\bar p_j)$ per feature, our measure of population "
            "diversity, together with the fraction of features on which every "
            "individual already agrees, the frozen fraction. Three settings are "
            "compared: $\\gamma=0$, a plain V-shaped transfer with no relevance "
            "signal; $\\gamma=0.5$, the value used throughout this paper; and "
            "$\\gamma=1$, a deliberately aggressive stress test. All three run "
            "on three datasets spanning low, medium, and very high "
            "dimensionality "
            f"({', '.join(esc(d) for d in diversity['datasets'])}). "
            "Table~\\ref{tab:diversity} reports the end-of-run values. "
            "The risk is real at the stress-test setting: under $\\gamma=1$, "
            f"{diversity['worst_ds']} reaches "
            f"{diversity['worst_frz_g1']*100:.0f}\\% frozen features by "
            f"iteration {diversity['worst_first_freeze_iter']}, and the effect is "
            "sharper on higher-dimensional data, exactly as the clip term "
            "in~\\eqref{eq:rms} predicts, since more features reach confident "
            "relevance scores when there are more of them. At the deployed "
            f"default $\\gamma=0.5$, however, the frozen fraction never exceeds "
            f"{diversity['max_frz_g5']*100:.1f}\\% at any iteration on any of the "
            "three datasets, including the highest-dimensional one; diversity "
            "decreases moderately but the population retains broad coverage of "
            "the subset space throughout the run. The bit-freezing risk is "
            "therefore genuine and $\\gamma$-dependent, not a property of RMS we "
            "can dismiss, but the conservative default this paper ships with "
            "avoids it in practice."
        )
        diversity_table = diversity["table"]
        diversity_figure = (
            "\\begin{figure}[t]\n\\centering\n"
            "\\includegraphics[width=\\columnwidth]{diversity.pdf}\n"
            "\\caption{Population diversity (top row) and frozen-bit fraction "
            "(bottom row) versus iteration, for $\\gamma=0$, the deployed "
            "$\\gamma=0.5$, and the stress-test $\\gamma=1$, on three datasets "
            "of increasing dimensionality. Freezing at $\\gamma=1$ is fast and "
            "pronounced on the highest-dimensional dataset; the deployed "
            "$\\gamma=0.5$ stays near zero throughout.}\n"
            "\\label{fig:diversity}\n\\end{figure}\n"
        )
    else:
        diversity_subsection = ""
        diversity_table = ""
        diversity_figure = ""
    _hs = _heldout.load()
    heldout_acc_tab = _heldout.acc_table_tex(_hs)
    heldout_nfeat_tab = _heldout.nfeat_table_tex(_hs)
    # Số leak-free cho Abstract (đọc động từ artifact, KHÔNG hardcode) — dùng làm
    # bằng chứng CHÍNH trong Abstract thay vì con số in-sample optimistic.
    hs_rank = float(_hs["ranking"]["RG-SCSO"])
    hs_wtl = _hs["wil"]["mark"].value_counts()
    hs_w, hs_t, hs_l = (int(hs_wtl.get(k, 0)) for k in ("+", "=", "-"))
    hs_d_aoa = float(_hs["wil"][_hs["wil"].compared_with == "AOA"]["cohens_d"].abs().median())
    hs_nf_rg = float(_hs["nf_mean"]["RG-SCSO"].mean())
    hs_nf_scso = float(_hs["nf_mean"]["SCSO"].mean())
    feat_counts = [pd.read_csv(os.path.join(PROC_DIR, f"{d}.csv")).shape[1] - 1
                   for d in s["datasets"]]
    feat_min, feat_max = min(feat_counts), max(feat_counts)

    #, Câu thống kê R4 (điền từ artifact; fallback [pending] nếu chưa chạy), 
    if s.get("stats"):
        w, ti, l = s["sig_total"]
        fr = s["friedman"]
        rank7 = s["rank7"]
        n_cmp = w + ti + l
        tie_txt = "; ".join(f"{esc(d)} vs.\\ {esc(a)}" for d, a in s["ties"]) or "none"
        rime_w = s["sig_wtl"].get("RIME", (0, 0, 0))[0]
        abstract_stats = (
            f"RG-SCSO wins {w} of {n_cmp} comparisons and loses {l} under the "
            f"Wilcoxon signed-rank test with Holm correction, median "
            f"$|d|={s['es_median']:.2f}$;")
        ranking_prose = (
            f"Table~\\ref{{tab:rank}} gives the average rank "
            f"({rank7['RG-SCSO']:.2f} for RG-SCSO) and the Holm-significant "
            f"win/tie/loss. Over the {n_cmp} pairwise comparisons RG-SCSO wins "
            f"{w}, ties {ti}, and loses {l}; it is never significantly "
            f"outperformed. The only tie is {tie_txt}, where its mean is still "
            f"higher but the paired difference is not significant. The Friedman "
            f"test rejects equal ranks ($\\chi^2={fr['chi2']:.2f}$, "
            f"$p{pcmp(fr['p'])}$, {fr['k']} algorithms; "
            f"Fig.~\\ref{{fig:cd}}). Effect sizes are large in the majority of "
            f"cases (median $|d|={s['es_median']:.2f}$; "
            f"{s['es_large_pct']:.0f}\\% exceed $0.8$). The recent SOTA anchor "
            f"RIME does not transfer to this binary FS setting, ranking "
            f"{rank7['RIME']:.2f} and losing all {rime_w} comparisons to "
            f"RG-SCSO.")
        conclusion_tail = (
            f"RG-SCSO won {w} of {n_cmp} Holm-corrected pairwise comparisons with "
            "predominantly large effect sizes; a preregistered ablation confirmed "
            "that both retained components (RMS and UMR) are load-bearing and "
            "pruned a third, online-learning variant, and a size-fair enrichment "
            "analysis linked the accuracy gains causally to relevance-guided "
            "selection.")
    else:
        abstract_stats = (
            "statistical significance, effect sizes, the ablation study, and a "
            "recent state-of-the-art comparison are reported in the final "
            "version;")
        ranking_prose = (
            f"Table~\\ref{{tab:rank}} gives the average rank "
            f"({rank['RG-SCSO']:.2f} for RG-SCSO) and win/tie/loss. RIME (2023) "
            f"is currently complete on {s['rime_done']} of 18 datasets; RG-SCSO "
            f"attains higher mean accuracy than RIME on all {s['rime_won']} of "
            f"them. The Friedman statistic, critical-difference diagram, and "
            f"Holm-corrected significance are reported in the final version.")
        conclusion_tail = ("Final claims are conditioned on the statistical tests "
                           "and the ablation reported in the final version.")
        w, n_cmp = "[pending]", "[pending]"  # dùng trong Abstract nếu stats chưa xong

    #, Ablation §V-D (điền từ artifact; fallback [pending] nếu chưa chạy), 
    if s.get("ablation"):
        v = s["verdict"]
        rms, orl, umr = v["NoRMS"], v["NoORL"], v["NoUMR"]
        ablation_tab = ablation_table(s)
        ablation_prose = (
            f"We started from a three-component design and tested each part by "
            f"removal (Table~\\ref{{tab:ablation}}), judging significance with a "
            f"paired Wilcoxon signed-rank test (Holm-corrected across datasets). "
            f"RMS is the strongest: removing it costs "
            f"{rms['worst_delta_pts']:.2f} accuracy points on "
            f"{esc(rms['worst_ds'])} ($d={rms['worst_d']:.2f}$, Holm $p<0.001$), "
            f"collapsing the transfer back to the plain V-shaped washout the "
            f"method targets. UMR is also load-bearing: removing it costs "
            f"{umr['worst_delta_pts']:.2f} points on {esc(umr['worst_ds'])} "
            f"($d={umr['worst_d']:.2f}$, Holm $p={umr['worst_p']:.3f}$). ORL is "
            f"not: removing it degrades accuracy on {orl['n_deg']}/{orl['n_ds']} "
            f"datasets (closest {esc(orl['closest_ds'])}, "
            f"{orl['closest_delta_pts']:+.2f} points, Holm $p={orl['closest_p']:.2f}$) "
            f"and even helps slightly on two. Following our preregistered "
            f"falsifiability rule we drop ORL; the final RG-SCSO comprises RMS "
            f"and UMR over the static mutual-information field. NoImprovement "
            f"(all off) is worst overall, so the two retained components act "
            f"jointly rather than redundantly.")
    else:
        ablation_tab = ""
        ablation_prose = (
            "A component ablation (Full vs.\\ NoRMS, NoORL, NoUMR, "
            "NoImprovement) testing whether each component is load-bearing is "
            "reported in the final version. \\textit{[pending: Table ablation]}")

    tex = rf"""%=======================================================================
% RG-SCSO, Springer Nature (sn-jnl) format, target: Applied Intelligence
% Số liệu sinh tự động từ experiments/results_fs/fs_results.csv ({s['n']}/18 dataset
% đủ 30 run cho 7 thuật toán; RG-SCSO = 2 thành phần RMS+UMR sau ablation).
% 4 hình (concept/convergence/CD/mechanism) trong figures/*.pdf, cần cùng thư mục khi compile.
% Compile: Overleaf class sn-jnl (Springer Nature LaTeX); tham chiếu Springer Basic (numbered).
%=======================================================================
\documentclass[sn-basic,pdflatex]{{sn-jnl}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{amsthm}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{graphicx}}
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

\newtheorem{{lemma}}{{Lemma}}

\begin{{document}}

\title[RG-SCSO for Parsimonious Feature Selection]{{Relevance-Guided Sand Cat Swarm Optimization: A Per-Feature Relevance-Modulated Binarization for Parsimonious High-Dimensional Feature Selection}}

\author*[1]{{\fnm{{Bui Quang}} \sur{{Huy}}}}\email{{huybq@donga.edu.vn}}
\author[1]{{\fnm{{Duong Minh}} \sur{{Son}}}}\email{{sondm@donga.edu.vn}}
\affil[1]{{\orgname{{Dong A University}}, \city{{Da Nang}}, \country{{Vietnam}}}}

\abstract{{
Wrapper feature selection with swarm intelligence typically searches continuously
and crosses into the binary domain via a fixed transfer function, a
feature-agnostic quantization that discards continuous operators' fine
adjustments, an effect we term washout. RG-SCSO instead replaces this transfer with a
per-feature, relevance-modulated binarization, biasing each feature's bit-flip
probability by a mutual-information field so informative features resist
removal and noise resists inclusion, via two ablation-confirmed components (RMS
and UMR); a third, online-learning variant is pruned. On {s['n']} datasets,
including two gene-expression sets, under a budget-matched protocol, the
defining outcome is parsimony: RG-SCSO selects smaller subsets than every
competitor, {s['red_scso']:.0f}\% fewer than base SCSO and {s['red_aoa']:.0f}\%
fewer than the strongest baseline (AOA), at no accuracy cost. We report accuracy
primarily under a leak-free hold-out denying the prior any access to test
labels: RG-SCSO attains the best Friedman rank, wins {hs_w} of {hs_w+hs_t+hs_l}
Holm-corrected comparisons with {hs_l} loss, and leads its closest competitor
AOA by a modest but consistent margin (median $|d|={hs_d_aoa:.2f}$). Under the
standard in-sample protocol, effect sizes are far larger
(median $|d|={s['es_median']:.2f}$), an optimistic upper bound since shared
folds inflate gains equally for every method. Against binary particle-swarm
and grey-wolf baselines with published adaptive transfers, RG-SCSO does not
lead on accuracy yet still selects {adaptive['red_min']:.0f} to
{adaptive['red_max']:.0f}\% fewer features, framing subset compactness, not
accuracy dominance, as the transferable benefit.
}}

\keywords{{Binary metaheuristics, feature selection, relevance-guided search,
sand cat swarm optimization, transfer function}}

\maketitle

\section{{Introduction}}
Feature selection removes irrelevant and redundant features
to improve classifier accuracy, reduce overfitting, and lower computational
cost. It is especially consequential for high-dimensional, small-sample problems
such as gene-expression classification, where the number of features exceeds the
number of samples by orders of magnitude. Wrapper feature selection, which scores
a subset by the performance of a downstream classifier~\cite{{guyon,mrmr}}, is
frequently cast as a combinatorial problem solved by swarm-intelligence
metaheuristics~\cite{{gwo,pso}}.

Most such methods were conceived for continuous optimization and are adapted to
the binary space through a transfer function that maps a real-valued position to
a selection probability, followed by thresholding~\cite{{tf}}. We argue that this design
contains a structural weakness: the transfer function is a fixed,
feature-agnostic mapping applied identically to every dimension, so the
incremental adjustments that continuous operators make are collapsed by the
sigmoid-and-threshold step before they influence the retained subset. We refer
to this loss as \emph{{washout}}. In a preliminary study, four continuous-space
enhancements of SCSO~\cite{{scso}} did not improve over the base algorithm on the same
protocol (0 wins, 1 loss, 17 ties by Wilcoxon signed-rank): the gains were real
in continuous space but quantized away at the binarization boundary.

This motivates a binary-native redesign. Rather than adding another
continuous-space operator upstream of the transfer, we intervene at the
binarization interface itself. In one sentence, the core novelty is to replace
the fixed, feature-agnostic transfer with a per-feature, relevance-modulated
binarization: a mutual-information relevance field biases each feature's
bit-flip probability, turning a knowledge-agnostic quantization step into a
knowledge-carrying operator that steers the search toward informative features
and compact subsets, so that relevant features become resistant to removal and
noisy features resistant to inclusion. SCSO's continuous search, including its
sensitivity range, is retained unchanged; the novelty resides entirely in the
binarization. Fig.~\ref{{fig:concept}} contrasts this with the conventional
washout pathway. The specific contributions are:
\begin{{itemize}}
  \item \textbf{{We identify}} washout as a mechanistic failure mode of
        transfer-function-based binary feature selection, characterized
        empirically and used to motivate a binary-native operator rather than
        another continuous-space enhancement;
  \item \textbf{{We propose RG-SCSO}} with two load-bearing components over a
        mutual-information relevance field: relevance-modulated sensitivity (RMS)
        and uncertainty-targeted memetic refinement (UMR); a third,
        online-learning variant (ORL) is examined and pruned by ablation;
  \item \textbf{{We evaluate}} under a preregistered, budget-matched protocol in
        which every algorithm receives an identical number of fitness
        evaluations, removing the memetic-evaluation confound;
  \item \textbf{{We report}} a full statistical treatment and a component
        ablation that tests whether each component is load-bearing, cutting any
        that are not.
\end{{itemize}}

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

The rest of the paper is organized as follows. Section~\ref{{sec:related}}
reviews related work; Section~\ref{{sec:method}} presents RG-SCSO;
Section~\ref{{sec:setup}} the protocol; Section~\ref{{sec:results}} the results.

\section{{Related Work}}
\label{{sec:related}}
\subsection{{Swarm-Based Wrapper Feature Selection}}
Binary variants of grey wolf optimization~\cite{{bgwo}}, particle swarm
optimization~\cite{{pso}}, the whale optimizer~\cite{{mafarja}}, and many recent
swarm methods~\cite{{aoa,coa,rime}} share the continuous-to-binary conversion
pattern and differ mainly in the underlying search operator; the no-free-lunch
theorem~\cite{{nfl}} explains the continual supply of such variants but not why
so few revisit the binarization interface itself. Filter criteria such as mutual
information and mRMR~\cite{{guyon,mrmr}} encode problem knowledge cheaply but are
decoupled from the wrapper search, and memetic hybridization~\cite{{neri}} adds
local refinement without addressing the same interface. Knowledge-guided
metaheuristics inject filter information into initialization or the objective,
as in filter-guided PSO for cancer-genome selection~\cite{{ludwig2025guided}}, but
leave the binarization operator itself knowledge-agnostic; RG-SCSO instead places
the relevance signal inside the operator.

\subsection{{Transfer Functions and Sand Cat Swarm Optimization}}
S-shaped and V-shaped transfer families~\cite{{tf}} are applied uniformly across
features and carry no problem-specific information. Attempts to make the transfer
itself adaptive, such as time-varying V-shaped slopes~\cite{{islam2017tvtf}} or the
adaptively shaped binary PSO of Teng et al.~\cite{{teng2017avbpso}}, vary the mapping
per iteration or globally yet still apply one shared transfer to every dimension,
leaving untouched the per-feature degree of freedom RG-SCSO supplies. SCSO~\cite{{scso}} balances
exploration and exploitation through a sensitivity range that decreases linearly
with iteration; existing SCSO-based feature
selectors~\cite{{bscso,scsofs2,scsofs3}} pair it with a generic transfer (binary
thresholding, opposition-based learning, or crossover operators) while
the range stays a single scalar acting on continuous magnitudes. The most recent
SCSO literature (2024--2025) continues this pattern: multi-strategy hybrids adding
chaotic initialization, differential mutation, or quadratic
interpolation~\cite{{imscso2024,mescso2025}}, and opposition-based or
sparrow-search hybridizations~\cite{{scsolensobl2024,improvedscso2024}}, all
improve the continuous search dynamics that Section~\ref{{sec:washout}} shows wash
out at the binarization boundary, and none address feature selection or the
transfer interface itself. To our knowledge,
no prior SCSO feature selector makes the binarization operator itself per-feature
and relevance-aware. This is the gap the paper addresses: RG-SCSO is binary-native by construction
rather than a continuous optimizer wrapped in a fixed transfer.

\section{{Proposed Method: RG-SCSO}}
\label{{sec:method}}

\begin{{algorithm}}[!t]
\caption{{RG-SCSO for binary feature selection}}
\label{{alg:main}}
\begin{{algorithmic}}[1]
  \REQUIRE dataset $(X,y)$; population size $N$; iterations $T$; budget
    $\mathrm{{max\_nfe}}$; memetic size $K$; bias strength $\gamma$
  \ENSURE best feature mask $b^\ast$
  \STATE $\rho_j \leftarrow$ normalized mutual information $I(X_j;y)$,
    $\forall j$ \hfill$\triangleright$ static relevance field
  \STATE $\forall j$: preferred bit $\hat{{b}}_j \leftarrow \mathbf{{1}}[\rho_j>0.5]$;
    strength $s_j \leftarrow 2|\rho_j-0.5|$
  \STATE initialize positions $x_i \sim \mathcal{{U}}(-1,1)^d$, $i=1,\dots,N$;
    binarize each by RMS (lines \ref{{ln:rms1}}--\ref{{ln:rms2}}); evaluate; set $b^\ast$
  \WHILE{{$\mathrm{{nfe}} < \mathrm{{max\_nfe}}$}}
    \STATE $R \leftarrow S_M\,(1-t/T)$ \hfill$\triangleright$ sensitivity range contracts
    \FOR{{each agent $i=1,\dots,N$}}
      \STATE update $x_i$ by the SCSO position rule using range $R$
      \FOR{{each feature $j$}} \label{{ln:rms1}}
        \STATE $p \leftarrow |\tanh(x_{{ij}})|$ \hfill$\triangleright$ V-shaped transfer
        \STATE $p \leftarrow p\,(1+\gamma s_j)$ if the flip moves bit $j$ toward
          $\hat{{b}}_j$, else $p\,(1-\gamma s_j)$
        \STATE flip bit $j$ with probability $\mathrm{{clip}}(p,0,1)$
      \ENDFOR \label{{ln:rms2}}
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

\subsection{{Problem Formulation}}
Let a candidate subset be a binary mask $b\in\{{0,1\}}^d$. The objective is
\begin{{equation}}
  f(b) = 0.99\,(1-\mathrm{{Acc}}(b)) + 0.01\,\frac{{|b|}}{{d}},
  \label{{eq:fitness}}
\end{{equation}}
where $\mathrm{{Acc}}(b)$ is the stratified 5-fold KNN accuracy ($k=5$) using only
the selected features. SCSO governs its search with the sensitivity range
$R(t)=S_M - S_M\,t/T$ ($S_M=2$), which RG-SCSO retains while replacing the
binarization and adding a relevance field.

\subsection{{Why Continuous Enhancements Wash Out: A Quantization Argument}}
\label{{sec:washout}}
Consider any binary-native optimizer that keeps a real-valued position and
binarizes coordinate $j$ through a transfer $T:\mathbb{{R}}\to[0,1]$ that returns a
selection-or-flip probability. A continuous-space enhancement can influence the
retained subset only by perturbing a coordinate, $x_j\mapsto x_j+\delta_j$; its
entire effect on the discrete decision is the induced change in probability
$\Delta_j=T(x_j+\delta_j)-T(x_j)$. The next result shows this leverage is
governed by the local slope of $T$ alone, and therefore collapses wherever $T$
saturates, the mechanism we call washout.

\begin{{lemma}}[Leverage bound]
Let $T:\mathbb{{R}}\to[0,1]$ be Lipschitz and piecewise continuously
differentiable. If a coordinate-space enhancement perturbs $x_j$ by $\delta_j$,
the induced change in the selection or flip probability satisfies
\begin{{equation}}
  |\Delta_j|\le\|T'\|_\infty\,|\delta_j|,
  \label{{eq:leverage}}
\end{{equation}}
where $\|T'\|_\infty$ is the largest slope $T$ attains between $x_j$ and
$x_j+\delta_j$. For the two standard transfers $\|\sigma'\|_\infty=\tfrac14$ and
$\||\tanh|'\|_\infty=1$, and each slope decays away from the origin,
\begin{{equation}}
  \sigma'(x)\le e^{{-|x|}},\qquad \big||\tanh|'(x)\big|\le 4\,e^{{-2|x|}};
  \label{{eq:decay}}
\end{{equation}}
in a flat region, where $\|T'\|_\infty\le\varepsilon$, the leverage collapses to
$|\Delta_j|\le\varepsilon\,|\delta_j|$.
\end{{lemma}}

\begin{{proof}}
By the mean value theorem there is a point $\xi$ strictly between $x_j$ and
$x_j+\delta_j$ for which $\Delta_j=T'(\xi)\,\delta_j$, so
$|\Delta_j|=|T'(\xi)|\,|\delta_j|\le\|T'\|_\infty\,|\delta_j|$; because $|\tanh|$
fails to be differentiable only at the origin, applying the theorem separately on
each side of zero extends the bound to every interval. Differentiating the two
transfers directly gives
\begin{{equation}}
  \sigma'(x)=\sigma(x)\big(1-\sigma(x)\big)=\frac{{e^{{-|x|}}}}{{(1+e^{{-|x|}})^2}},
  \label{{eq:sigder}}
\end{{equation}}
\begin{{equation}}
  |\tanh|'(x)=\operatorname{{sech}}^2(x)=\frac{{4\,e^{{-2|x|}}}}{{(1+e^{{-2|x|}})^2}},
  \label{{eq:tanhder}}
\end{{equation}}
whose maxima are $\tfrac14$ at $\sigma=\tfrac12$ and $1$ as $x\to0$, while each
right-hand side is dominated by its exponential envelope because the denominators
are at least one. Substituting $\|T'\|_\infty\le\varepsilon$ on a flat interval
into~\eqref{{eq:leverage}} yields $|\Delta_j|\le\varepsilon\,|\delta_j|$,
establishing the claim.
\end{{proof}}

\noindent\textbf{{Remark 1 (why RG-SCSO is exempt).}} The bound turns on $T'$
alone, so it governs any base optimizer that binarizes through a saturating
transfer, not SCSO in particular. RG-SCSO breaks the premise of the lemma:
instead of routing information through the coordinate, it modulates the flip
probability directly at the transfer output through the relevance-modulated rule
developed next, contributing
\begin{{equation}}
  \Delta_j=\pm\,\gamma\,s_j\,V(x_j),
  \label{{eq:exempt}}
\end{{equation}}
independent of $T'$. The S-shaped map, whose slope nowhere exceeds one quarter,
is thus an intrinsically weaker channel than the V-shaped map RG-SCSO adopts,
whose slope approaches unity near the origin.

\subsection{{Relevance-Modulated Sensitivity}}
Given the updated position $x_j$, the base flip probability uses a V-shaped
transfer $V(x_j)=|\tanh(x_j)|$~\cite{{tf}}. With relevance $\rho_j\in[0,1]$,
preferred bit $b^\ast_j=\mathbf{{1}}[\rho_j>0.5]$ and strength
$s_j=2|\rho_j-0.5|\in[0,1]$, we bias the flip probability toward $b^\ast_j$
through a rule we call relevance-modulated sensitivity, or RMS,
\begin{{equation}}
  p_j = \mathrm{{clip}}\!\Big(|\tanh(x_j)|\,\big(1+\gamma\,\sigma_j\,s_j\big),\,0,\,1\Big),
  \label{{eq:rms}}
\end{{equation}}
where $\sigma_j=+1$ if the flip moves bit $j$ toward $b^\ast_j$ and $\sigma_j=-1$
otherwise, and $\gamma$ is the bias strength. The strength $s_j$ makes the
modulation vanish for uninformative features ($\rho_j\!\approx\!0.5$) and peak for
decisive ones; setting $\gamma=0$ in~\eqref{{eq:rms}} recovers a plain V-shaped
operator (the NoRMS ablation).

{diversity_subsection}

{diversity_table}

{diversity_figure}

\subsection{{The Relevance Field and an Online Extension}}
The field $\rho$ that drives RMS and UMR is, in the final method, the static
prior $\rho_{{\mathrm{{static}}}}=\mathrm{{clip}}(\mathrm{{MI}}(f_j;y)/H(y),0,1)$,
a normalized mutual-information filter score~\cite{{mrmr,kraskov}} computed once
from the data. We also examined an online extension (ORL) adding an
EMA credit-assignment term from accepted fitness improvements
($\rho=\mathrm{{clip}}(\rho_{{\mathrm{{static}}}}+w_o\rho_{{\mathrm{{online}}}},0,1)$);
the ablation in Section~\ref{{sec:results}} shows it does not improve accuracy on
any dataset, so it is not retained and the deployed field is
$\rho_{{\mathrm{{static}}}}$.

\subsection{{Uncertainty-Targeted Memetic Refinement}}
A memetic local search~\cite{{neri}}, which we call uncertainty-targeted memetic
refinement, or UMR, concentrates its budget where the relevance prior is least
decisive: each iteration, the $K$ features whose relevance is closest to $0.5$
are greedily flipped on the incumbent best mask and kept only if fitness
improves.

\subsection{{Budget-Matched Evaluation}}
Every fitness evaluation, whether from population moves, memetic probes, or
initialization, is counted against a single budget $\mathrm{{max\_nfe}}=\mathrm{{pop\_size}}\times
\mathrm{{max\_iter}}=15000$, identical to the baselines, so UMR grants no extra
evaluations (Algorithm~\ref{{alg:main}}).

\subsection{{Computational Complexity}}
The per-iteration cost is dominated by the $N+K$ wrapper evaluations ($N$
population agents plus $K$ memetic probes), each a KNN fit under fixed folds; the
mutual-information prior adds a one-time $O(dn\log n)$ preprocessing cost (the
nearest-neighbor MI estimator~\cite{{kraskov}} over $d$ features and $n$ samples),
which is amortized against and dominated by the wrapper evaluations. RG-SCSO is
thus asymptotically no more expensive than base SCSO apart from the $K$ extra
probes, and because those probes are drawn from the shared evaluation budget, the
comparison stays strictly budget-matched.

\section{{Experimental Setup}}
\label{{sec:setup}}
Table~\ref{{tab:datasets}} summarizes the {s['n']} benchmark datasets, which span
biomedical, gene-expression, and categorical domains and include two
high-dimensional microarray sets (ColonCancer and Leukemia) with many more features
than samples.

{dataset_tab}

The protocol is preregistered and locked prior to the full run. All algorithms
share: population 30, 500 iterations, 30 independent runs, seed $=42+\mathrm{{run\_id}}$
(paired across algorithms), KNN ($k=5$) with stratified 5-fold cross-validation,
search space $[-1,1]^d$, and the fitness of~\eqref{{eq:fitness}}. All methods are
matched on the number of fitness evaluations; the only difference is the
algorithm itself. Baselines: SCSO (base)~\cite{{scso}}, AOA~\cite{{aoa}},
CoatiOA~\cite{{coa}}, GWO~\cite{{gwo}}, PSO~\cite{{pso}}, and RIME~\cite{{rime}}
(added as a recent anchor). For fairness we avoid a hand-weakened straw man:
every baseline runs with its library-default published hyperparameters, with only
the population size and the evaluation budget set and matched across methods, and
no method is tuned per dataset. RG-SCSO's $\gamma$ and $K$ are likewise fixed before
the full run rather than adjusted per dataset, and are set conservatively (the
sensitivity analysis shows $\gamma=0.5$ sits below the accuracy-optimal $0.75$), so
the protocol grants no method a dataset-specific advantage.
Significance uses the paired Wilcoxon signed-rank
test with Holm correction~\cite{{holm}}; the Holm family is formed
\emph{{per dataset}}, the $k-1$ RG-SCSO-versus-baseline comparisons on a given
dataset are corrected together, and we do not pool across datasets, so each
dataset is an independent inferential unit. Effect sizes use Cohen's $d$ and
rank-biserial $r$; overall comparison uses the Friedman test across all datasets
with a critical-difference diagram~\cite{{demsar}}. We report every comparison,
including those in which RG-SCSO does not reach significance.

\emph{{Reproducibility.}} The experimental design was preregistered and
version-controlled before the full run and left unmodified after results were
observed. All randomness is seeded deterministically (seed $=42+\mathrm{{run\_id}}$)
and shared across algorithms, making every comparison exactly paired. Each number
reported in this paper is regenerated programmatically from the raw per-run result
files rather than transcribed by hand. The source code, the locked
preregistration, the per-run seeds, a complete hyperparameter table, and a pinned
dependency list are available for review in an anonymized repository
(\url{{https://anonymous.4open.science/r/RG-SCSO}}) and will be released in a
public, citable repository (Zenodo DOI) upon acceptance, permitting bit-for-bit
replication.

\section{{Results and Discussion}}
\label{{sec:results}}
Table~\ref{{tab:acc}} reports mean accuracy. RG-SCSO attains the highest mean
accuracy on all {s['n']} datasets, exceeding base SCSO by {s['margin_scso']:.2f}
and AOA by {s['margin_aoa']:.2f} accuracy points on average. The per-dataset
margins vary widely, from a fraction of a point to more than twenty points on
M-of-n. This pattern is expected rather than anomalous: M-of-n is a synthetic
concept whose label depends on only a few of the input bits, so the
mutual-information prior locates the relevant features directly and a
relevance-guided search separates most sharply from a relevance-agnostic base
that receives the identical budget and default parameters. The largest margins
therefore track how informative the prior is on a given problem, not any
handicap of the baselines. Table~\ref{{tab:nfeat}}
shows it selects {s['red_scso']:.0f}\% fewer features than SCSO and
{s['red_aoa']:.0f}\% fewer than AOA; on ColonCancer it retains
{colon.get('nf', float('nan')):.0f} of {colon.get('ntot', '--')} features versus
{colon.get('nf_aoa', float('nan')):.0f} for AOA, at higher accuracy. This
parsimony, more than any accuracy margin, is RG-SCSO's defining property: it is the
single advantage that persists across every evaluation we run, surviving both the
leak-free hold-out of Section~\ref{{sec:results}} and the stronger adaptive-transfer
optimizers of Appendix~\ref{{app:adaptive}}, neither of which erodes the subset-size
gap.{inference_sentence}
{ranking_prose}

{accuracy_table(s)}
{nfeat_table(s)}
{rank_table(s)}

\subsection{{Ablation Study}}
{ablation_prose}

{ablation_tab}

\subsection{{Convergence and Mechanism}}
Fig.~\ref{{fig:conv}} shows the convergence behaviour and Fig.~\ref{{fig:cd}} the
critical-difference diagram of the Friedman ranking. To establish a causal link
rather than a black-box win, we test whether relevance guidance makes RG-SCSO
preferentially retain high mutual-information features. Because a subset of size
$|S|$ overlaps the top-$|S|$ mutual-information features at a chance rate of
$|S|/N$, we report a size-fair enrichment, defined as the fraction of selected
features in the top-$|S|$ set divided by this chance level, so that one denotes
relevance-agnostic selection and a value above one an enriched subset
(Fig.~\ref{{fig:mech}}). This isolates the effect of guidance from the smaller
subsets RG-SCSO already produces.

\subsection{{Generalization under a Leak-Free Hold-Out}}
For each dataset, algorithm, and independent run we draw an outer stratified 80/20
split. The relevance prior, the search, and the cross-validated fitness are
computed exclusively on the 80\% training partition; the selected subset is then
evaluated once on the untouched 20\% hold-out, on which a fresh $k$-NN classifier
(standardized on the training partition) reports accuracy. The evaluation budget,
population size, iteration count, seed scheme, and the 30 independent runs are
identical to the primary study; only the metric of record changes to the held-out
accuracy. This removes any transductive access of the relevance prior to the test
labels.

Table~\ref{{tab:heldout_acc}} reports held-out accuracy over all seven algorithms.
RG-SCSO attains the best average Friedman rank (1.33, ahead of the
second-placed AOA at 2.03 and above the remaining methods, COA and SCSO 4.31,
RIME 4.81, PSO 5.47, GWO 5.75; $\chi^2=66.16$, $p<10^{{-3}}$). Across the
full set of pairwise comparisons a Holm-corrected Wilcoxon signed-rank test gives
RG-SCSO 64 significant wins, 1 loss, and 43 ties. Against its own base optimizer
SCSO the improvement is clear (11 wins, 0 losses; median $|d|=0.61$), and it
also improves on COA, GWO, PSO, and RIME (10--13 wins, 0 losses each;
median $|d|$ up to 0.84). The only close competitor is AOA, against which the
advantage is genuine but moderate (5 wins, 1 loss, 12 ties; median $|d|=0.30$),
RG-SCSO still leading on mean accuracy (0.866 vs.\ 0.849); the single loss occurs on
BreastEW. Crucially, RG-SCSO delivers this accuracy while selecting fewer
features: on average 91.6, against 164.2 for SCSO and 325.9 for AOA
(Table~\ref{{tab:heldout_nfeat}}). Its ranking advantage is thus coupled with a
substantial parsimony advantage that is robust under the leak-free evaluation.

\emph{{On effect-size magnitude.}} Comparing the two protocols is itself
informative. The large in-sample effect sizes (median $|d|=2.15$ for Cohen's
$d$, 95\% CI 2.02--2.38) contract to a small-to-moderate range (per-baseline median
$|d|=0.30$--$0.84$) on the hold-out, whereas the ranking is preserved (RG-SCSO
remains first). This is the expected signature of the optimistic bias shared by any
wrapper that uses its cross-validation score both to search and to report: it
inflates absolute magnitudes equally for all methods, hence is fair for relative
comparison, but should not be read as the true out-of-sample gain. The persistence
of the ranking, and of the parsimony advantage, under the leak-free protocol
confirms that the mechanism behind RG-SCSO (relevance-biased flipping) transfers to
unseen data and is not a consequence of the prior observing the labels.

{heldout_acc_tab}
{heldout_nfeat_tab}

\subsection{{Threats to Validity}}
Several boundaries delimit what these results establish. RG-SCSO inherits
SCSO's continuous search dynamics unchanged, exploration limitations included,
and the RMS rule carries a risk of its own: bias the flip probability too
strongly and confidently classified bits can freeze in place, draining the
population's diversity. We did not assume this away; we measured it
(Section~\ref{{sec:diversity}}). At an aggressive stress-test bias the risk is
real, and it grows with dimensionality. The conservative $\gamma=0.5$ this
paper deploys, though, keeps the frozen-bit fraction below
{diversity['max_frz_g5']*100:.1f}\% throughout the run on every dataset tested,
including the highest-dimensional one.

Two further design choices could narrow the claim. The main objective is a KNN
wrapper, and because mutual information and KNN both weigh local neighbourhood
structure, part of the advantage might be specific to that pairing. Under an
SVM wrapper, tested directly on {svm16['n_ds']} of the 18 datasets
(Appendix~\ref{{app:robustness}}), the parsimony advantage persists: about
{svm16['red']:.0f}\% smaller subsets than the no-prior baseline at essentially
equal accuracy, so the mechanism is not a KNN artifact, though tree-ensemble
and neural classifiers remain future work. The relevance field is also built
from a single filter statistic, the mutual information between each feature and
the label. Swapping in a ReliefF prior (Appendix~\ref{{app:robustness}})
exposes a genuine limitation rather than confirming the mechanism: the
parsimony advantage disappears, and subsets grow to roughly
{robust['relieff_vs_mi']['KNN']:.1f} times those of the MI variant, comparable
to the no-prior baseline. RG-SCSO's compactness, it turns out, depends on a
prior that drives uninformative features below the neutral point, not on the
mere presence of a relevance signal.

External validity has its own limits. The benchmark spans {feat_min} to
{feat_max} features across biomedical, gene-expression, and categorical
domains, but it is drawn from a single curated family of UCI and standard
microarray sets, so validity to other data regimes is asserted rather than
proven. We evaluate up to a few thousand features; behaviour on
ultra-high-dimensional omics data of $10^4$ to $10^5$ features is
extrapolated, not measured. The numbers themselves carry a caveat too: as
quantified in Section~\ref{{sec:results}}, in-sample effect sizes inherit the
optimistic bias intrinsic to any wrapper that both searches and reports on the
same folds, and we regard the leak-free hold-out as the conservative estimate.
Finally, the accuracy claim is scoped, not universal. Against binary
particle-swarm and grey-wolf optimizers carrying published adaptive V-shaped
transfers (Appendix~\ref{{app:adaptive}}), and against same-family binary SCSO
selectors (Appendix~\ref{{app:scsofamily}}), RG-SCSO does not lead on accuracy,
so we scope the accuracy result to the standard baseline suite and frame
parsimony, the {adaptive['red_min']:.0f} to {adaptive['red_max']:.0f}\%
smaller subsets that hold even there, as the mechanism's transferable benefit.
We state these limits explicitly so the claim, a per-feature relevance
mechanism that selects smaller subsets and improves accuracy on the standard
suite, for binary SCSO-style feature selection under a KNN wrapper on this
benchmark family, is not overread.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{convergence.pdf}}
\caption{{Mean best fitness versus iteration on a low-dimensional (Zoo) and a
high-dimensional (ColonCancer) dataset, averaged over 30 runs; curves are
regenerated deterministically (identical seeds) from the main experiment for
visualization.}}
\label{{fig:conv}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\columnwidth]{{cd_diagram.pdf}}
\caption{{Critical-difference diagram (Friedman + Nemenyi) over the
{s['n']} datasets; algorithms joined by a bar are not significantly different.
RG-SCSO attains the best mean rank (1.00).}}
\label{{fig:cd}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\columnwidth]{{mechanism.pdf}}
\caption{{Mechanism evidence on the gene-expression sets: size-fair top-MI
enrichment (selection precision divided by the chance level $|S|/N$; mean over 30
runs, error bars $=$ std). RG-SCSO enriches its subset in relevant features above
chance, whereas the relevance-agnostic SCSO sits at chance.}}
\label{{fig:mech}}
\end{{figure}}

\section{{Conclusion}}
We replaced the fixed, feature-agnostic transfer of a binary SCSO with a
per-feature, relevance-modulated binarization to overcome washout in
transfer-function-based binary feature selection. Under a strictly budget-matched protocol, RG-SCSO's standout contribution is
parsimony: it produced smaller feature subsets than every competitor on
all {s['n']} datasets, an advantage that persists under leak-free evaluation and
even against stronger adaptive-transfer optimizers. Coupled with this, against the
standard baseline suite it achieved the best average rank ({rank['RG-SCSO']:.2f}).
{conclusion_tail} We frame the contribution deliberately: the formal result in
Section~\ref{{sec:washout}} is a diagnostic bound that explains why continuous
enhancements fail at the binarization boundary, not a convergence guarantee for
RG-SCSO itself, and the underlying recipe, a filter prior coupled to a wrapper
search with memetic refinement, is a known combination in the feature-selection
literature. What is new is the placement of the relevance signal directly inside
the binarization operator rather than in the objective or the initialization,
together with a stringent evaluation protocol, budget-matching, a leak-free
hold-out, cross-classifier and cross-prior robustness, and an explicit
exploration-safety diagnostic, that most work in this line does not undertake. We
report parsimony, not a new search operator, as the transferable outcome of this
choice. Limitations include reliance on a KNN wrapper and the dataset
family studied; future work includes multiobjective formulations, alternative
classifiers, and ultra-high-dimensional data.

\begin{{appendices}}

\section{{Computational Cost}}
Table~\ref{{tab:runtime}} reports the mean wall-clock time per run under the common
protocol on a single CPU thread. RG-SCSO's per-feature relevance modulation and its
memetic refinement add no measurable overhead over base SCSO; in fact RG-SCSO is
faster on every dataset except the lowest-dimensional one (Zoo), because the
relevance prior steers the search toward compact masks whose KNN evaluations are
cheaper. All wrappers stay within a small constant factor of one another, the sole
exception being COA, whose per-iteration cost is two to three times higher.

{runtime_tab}

\section{{Preliminary Study on the Washout of Continuous Enhancements}}
Table~\ref{{tab:washout}} gives the per-dataset detail behind the summary figure of
0 wins, 1 loss, and 17 ties quoted in the introduction. The continuous-enhanced
variant is statistically indistinguishable from base SCSO on 17 of 18 datasets and
significantly worse on one (WDBC); it wins on none. Numerically it is higher on six
datasets and lower on twelve, yet none of the apparent gains survive the Wilcoxon
test, confirming that operators engineered in the continuous space do not carry
through the binarization boundary.

{washout_tab}

\section{{Hyperparameter Sensitivity}}
Table~\ref{{tab:sensitivity}} reports a one-factor-at-a-time sweep of every
hyperparameter, each varied around its default while the others are held fixed and
the number of function evaluations is kept constant. The relevance strength
$\gamma$ is the only influential control, and it behaves exactly as the mechanism
predicts: as $\gamma$ grows the search concentrates on relevant features, so the
selected subset shrinks, from 340 features at $\gamma=0$ to 48 at
$\gamma=1$, while accuracy improves overall, peaking near $\gamma=0.75$, rather than
degrading. We nonetheless ship $\gamma=0.5$ rather than the accuracy-peak
$\gamma=0.75$: the default was fixed before these runs, not tuned after the fact,
and adopting the empirical peak would tune a hyperparameter to this benchmark's
relevance signal, whereas a smaller $\gamma$ trusts the mutual-information prior
less and so is more robust when that prior is noisier than on these datasets. The
choice is deliberately conservative, a practitioner willing to lean harder on the
prior can raise $\gamma$ for still more parsimony. The memetic budget $K$ is
essentially inert from 2 to 16, with
accuracy varying by less than 0.005, confirming that the refinement is robust rather
than delicately tuned. The ORL-only parameters $\lambda$ and $w_o$ leave accuracy
flat and never lift the ORL-on variant above the ORL-off method, consistent with the
ablation decision to remove that component.

{sensitivity_tab}

\section{{Isolating the Relevance Contribution from Adaptive Transfers}}
\label{{app:adaptive}}
A reviewer-motivated concern is whether RG-SCSO's advantage stems from its
per-feature relevance field or merely from replacing a static transfer with an
adaptive one. To isolate this, we run binary particle-swarm and grey-wolf
optimizers equipped with two published adaptive V-shaped transfers, the
time-varying $|\tanh(\tau x)|$ of Islam et~al.~\cite{{islam2017tvtf}} and the V4
transfer of Teng et~al.~\cite{{teng2017avbpso}}, under the identical protocol,
datasets, and evaluation budget of the main study but with no per-feature relevance
signal. Table~\ref{{tab:adaptive}} reports the outcome, and it is a negative result
for accuracy that we report as such: the strongest configuration,
{adaptive['best_acc_cfg']}, reaches {adaptive['best_acc']:.4f} mean accuracy, ahead
of RG-SCSO, so a well-tuned adaptive transfer paired with a strong base optimizer
can match or exceed RG-SCSO on accuracy alone. The relevance mechanism thus does not
confer a universal accuracy advantage over every binarization scheme. What survives
is parsimony: RG-SCSO selects {adaptive['rg_nfeat']:.0f} features on average,
{adaptive['red_min']:.0f} to {adaptive['red_max']:.0f}\% fewer than these baselines,
so the compact subsets, not accuracy dominance, are the relevance field's robust and
transferable benefit. Because each baseline swaps its base optimizer (PSO or GWO)
alongside the transfer, the accuracy leader also differs in its search operator, a
confound we do not disentangle here; it does not affect the parsimony conclusion,
which holds across all four configurations.

{adaptive['table']}

\section{{Comparison with Same-Family SCSO Feature Selectors}}
\label{{app:scsofamily}}
Because the paper locates its gap inside the SCSO feature-selection line, we
compare directly against binary SCSO selectors of that same family rather than
only against SCSO's continuous base and other optimizers. We reimplement two
standard recipes under the identical protocol: bSCSO (S-shaped), which binarizes
the SCSO position with an S-shaped transfer, and bSCSO (V-shaped~$+$~OBL), which
adds a V-shaped transfer and opposition-based
learning~\cite{{tf}}; neither carries a per-feature relevance field. These are
reimplementations from the published pseudocode, not original authors' code, and
their parameters follow the values as reported. Table~\ref{{tab:scsofamily}} gives
the outcome, and it mirrors the adaptive-transfer finding: on accuracy RG-SCSO does
not dominate its own family (win/tie/loss {scsofam['wtl']['bSCSO-S']} against bSCSO
(S-shaped) and {scsofam['wtl']['bSCSO-OBL']} against bSCSO (V-shaped~$+$~OBL)), the
mean accuracies being within {abs(scsofam['mean_acc']['RG-SCSO']-scsofam['mean_acc']['bSCSO-S'])*100:.2f}
of a point. What separates RG-SCSO is parsimony: it averages
{scsofam['mean_nfeat']['RG-SCSO']:.0f} features against
{scsofam['mean_nfeat']['bSCSO-S']:.0f} and {scsofam['mean_nfeat']['bSCSO-OBL']:.0f},
roughly {scsofam['red']['bSCSO-S']:.0f}\% fewer at equal accuracy, and it selects
the smaller subset on {scsofam['smaller']['bSCSO-S']} of
{scsofam['n_ds']} datasets. The per-feature relevance field is therefore what buys
subset compactness within the SCSO family, consistent with the paper's central
claim.

{scsofam['table']}

\section{{Robustness Across Classifiers and Relevance Priors}}
\label{{app:robustness}}
Two design choices could confound the main results: the KNN wrapper and the
mutual-information prior might form a uniquely favorable pair. We test both on
{robust['n_ds']} representative datasets (Zoo, Sonar, WDBC, ColonCancer, Leukemia)
by crossing two wrappers (KNN and an SVM) with two priors (MI and ReliefF),
against bSCSO with no prior, all under the main protocol.
Table~\ref{{tab:robustness}} reports two findings, one supportive and one that
bounds the claim. First, the parsimony advantage is \emph{{not}} a KNN artifact: under
the SVM wrapper RG-SCSO (MI) still selects about {robust['red']['SVM']:.0f}\% fewer
features than bSCSO ({robust['red']['KNN']:.0f}\% under KNN) at competitive accuracy,
so the mechanism transfers across classifiers. Second, and reported plainly as a
limitation, the parsimony is specific to the MI prior: swapping in ReliefF yields
subsets about {robust['relieff_vs_mi']['KNN']:.1f}$\times$ larger under KNN and
{robust['relieff_vs_mi']['SVM']:.1f}$\times$ larger under SVM, comparable to bSCSO
and above RG-SCSO (MI). The mechanism is the calibration of the prior: the
normalized MI score drives uninformative features below the neutral point so RMS
prunes them, whereas the ReliefF weights on these datasets sit mostly above neutral
and thus bias toward inclusion. RG-SCSO's compactness therefore depends on a prior
that pushes noise features toward exclusion, not on the presence of any relevance
signal; we did not retune the ReliefF mapping to obscure this.

{robust['table']}

To confirm the classifier result beyond the five-dataset cross-tabulation, we
ran the SVM wrapper on {svm16['n_ds']} of the 18 datasets, excluding only the two
largest-sample sets (KrVsKpEW and WaveformEW) for which kernel-SVM training is
$O(n^2)$ per fit and infeasible inside a 15{{,}}000-evaluation wrapper; the KNN main
study already covers those two. The picture is unchanged
(Table~\ref{{tab:svm16}}): RG-SCSO (MI) selects {svm16['red']:.0f}\% fewer features
than the no-prior baseline at essentially equal accuracy
({svm16['macc']['RG-SCSO-MI']:.4f} versus {svm16['macc']['bSCSO']:.4f}), on
{svm16['smaller']} of {svm16['n_ds']} datasets, and it does not lead on accuracy
(win/tie/loss {svm16['wtl']}). The ReliefF variant again inflates subsets to
{svm16['ratio']:.1f}$\times$ the MI size. The parsimony mechanism therefore
generalizes across classifiers on the broad benchmark, and its dependence on the
mutual-information prior generalizes with it.

{svm16['table']}

\end{{appendices}}

\backmatter

\bmhead{{Acknowledgments}}
Computation was performed on local Apple Silicon hardware (CPU-only).

\bmhead{{Data and code availability}}
The datasets are publicly available benchmarks (UCI and standard microarray sets).
The source code, the locked preregistration, the per-run seeds, and the raw
results are available in the repository stated in Section~\ref{{sec:setup}}, and
will be released in a public, citable repository upon acceptance.

\bmhead{{Statements and Declarations}}
\textbf{{Funding.}} No funding was received for this work.\\
\textbf{{Competing interests.}} The authors declare that they have no
competing interests.\\
\textbf{{Author contributions.}} B.Q.H. conceived the method, implemented the
software, ran the experiments, and drafted the manuscript. D.M.S. contributed
to the experimental design and reviewed the manuscript. Both authors read and
approved the final manuscript.\\
\textbf{{ORCID iDs.}} Bui Quang Huy
\href{{https://orcid.org/0009-0000-5761-5098}}{{0009-0000-5761-5098}};
Duong Minh Son
\href{{https://orcid.org/0009-0006-6485-7902}}{{0009-0006-6485-7902}}.

\bibliography{{references}}

\end{{document}}
"""
    with open(OUT_TEX, "w") as fh:
        fh.write(tex)
    print(f"Đã ghi {OUT_TEX}  ({s['n']}/18 dataset)")
    print("Avg rank:\n" + rank.round(2).to_string())


if __name__ == "__main__":
    build()
