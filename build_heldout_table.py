"""Sinh bảng generalization held-out (R3b) cho paper — SINGLE SOURCE OF TRUTH.

Đọc CHỈ các artifact tự sinh, KHÔNG gõ tay số nào:
    experiments/results_fs_heldout/fs_heldout_results.csv   (raw, 30 run/cell)
    experiments/results_fs_heldout/friedman_ranking.csv     (rank held-out)
    experiments/results_fs_heldout/friedman_summary.csv     (chi2, p)
    experiments/results_fs_heldout/wilcoxon_vs_rgscso.csv    (Holm marks, d)

Xuất:
    experiments/results_fs_heldout/generalization_acc.tex   (bảng accuracy held-out, IEEEtran table*)
    experiments/results_fs_heldout/generalization_nfeat.tex (bảng #features held-out)
    HELDOUT_REVIEW.md  (preview markdown + đoạn disclosure để user DUYỆT trước khi chèn docx)

DYNAMIC theo số thuật toán CÓ trong CSV: chạy khi mới 3 algo (RG/SCSO/AOA) sẽ ra bảng 3 cột;
sau khi Stage 2 xong (đủ 7) chạy lại → bảng 7 cột tự động. RG-SCSO luôn cột đầu, còn lại xếp
theo rank Friedman (tốt→kém). KHÔNG diễn giải/tô hồng — chỉ trình bày số + dấu thống kê.

Chạy:  python build_heldout_table.py
"""

from __future__ import annotations

import os

import pandas as pd

HELDOUT_DIR = os.path.join("experiments", "results_fs_heldout")
RESULTS = os.path.join(HELDOUT_DIR, "fs_heldout_results.csv")
RANKING = os.path.join(HELDOUT_DIR, "friedman_ranking.csv")
FRIEDMAN = os.path.join(HELDOUT_DIR, "friedman_summary.csv")
WILCOXON = os.path.join(HELDOUT_DIR, "wilcoxon_vs_rgscso.csv")
TARGET = "RG-SCSO"


def _esc(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&")


def load() -> dict:
    df = pd.read_csv(RESULTS)
    # Chỉ giữ thuật toán ĐÃ đủ 30 run trên MỌI dataset (18×30). Khi Stage 2 đang
    # append dở, algo chưa xong bị loại → preview luôn sạch; tự thành 7 khi xong.
    n_ds = df["dataset"].nunique()
    complete = [a for a, g in df.groupby("algorithm")
                if (g.groupby("dataset").size() == 30).sum() == n_ds]
    df = df[df["algorithm"].isin(complete)]

    ranking = pd.read_csv(RANKING).set_index("algorithm")["avg_rank"]
    ranking = ranking[ranking.index.isin(complete)]
    # Thứ tự cột: RG-SCSO trước, phần còn lại theo rank held-out (tốt→kém).
    others = [a for a in ranking.sort_values().index if a != TARGET]
    algos = [TARGET] + others

    acc_mean = df.pivot_table(index="dataset", columns="algorithm", values="heldout_accuracy", aggfunc="mean")
    acc_std = df.pivot_table(index="dataset", columns="algorithm", values="heldout_accuracy", aggfunc="std")
    nf_mean = df.pivot_table(index="dataset", columns="algorithm", values="n_selected_features", aggfunc="mean")
    ntot = df.groupby("dataset")["n_total_features"].first()

    stats = {}
    if os.path.exists(FRIEDMAN):
        fr = pd.read_csv(FRIEDMAN).iloc[0]
        stats["friedman_chi2"] = float(fr["statistic"])
        stats["friedman_p"] = float(fr["p_value"])
    wil = pd.read_csv(WILCOXON) if os.path.exists(WILCOXON) else pd.DataFrame()

    return {
        "df": df,
        "algos": algos,
        "datasets": sorted(acc_mean.index),
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "nf_mean": nf_mean,
        "ntot": ntot,
        "ranking": ranking,
        "stats": stats,
        "wil": wil,
    }


def acc_table_tex(s: dict) -> str:
    algos = s["algos"]
    cols = "l" + "c" * (len(algos) + 1)
    head = " & ".join(["Dataset", "\\#F"] + [_esc(a) for a in algos])
    lines = []
    for ds in s["datasets"]:
        row = s["acc_mean"].loc[ds]
        best = row.max()
        cells = [_esc(ds), str(int(s["ntot"][ds]))]
        for a in algos:
            m, sd = s["acc_mean"].loc[ds, a], s["acc_std"].loc[ds, a]
            val = f"{m:.4f}$\\pm${sd:.3f}"
            if abs(m - best) < 1e-9:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (
        "\\begin{sidewaystable}[t]\n\\centering\n"
        "\\caption{Held-out generalization: mean $\\pm$ std accuracy on the outer 20\\% "
        "hold-out over 30 runs (relevance prior, search, and CV fitness fit on the 80\\% "
        "training split only). \\#F = total features. \\textbf{Bold} = best per dataset. "
        "Higher is better ($\\uparrow$).}\n"
        "\\label{tab:heldout_acc}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{sidewaystable}}\n"
    )


def nfeat_table_tex(s: dict) -> str:
    algos = s["algos"]
    cols = "l" + "c" * len(algos)
    head = " & ".join(["Dataset"] + [_esc(a) for a in algos])
    lines = []
    for ds in s["datasets"]:
        row = s["nf_mean"].loc[ds]
        # Bold at the REPORTED precision (1 decimal): rows that display the same
        # fewest value are tied and both bolded, so the table never shows two
        # identical figures with inconsistent emphasis.
        least = round(row.min(), 1)
        cells = [_esc(ds)]
        for a in algos:
            v = s["nf_mean"].loc[ds, a]
            val = f"{v:.1f}"
            if round(v, 1) == least:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Held-out setting: mean number of selected features over 30 runs. "
        "\\textbf{Bold} = fewest ($\\downarrow$).}\n"
        "\\label{tab:heldout_nfeat}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def markdown_preview(s: dict) -> str:
    algos = s["algos"]
    md = []
    n_algo = len(algos)
    complete = "ĐỦ 7 thuật toán" if n_algo >= 7 else f"MỚI {n_algo} thuật toán (Stage 2 đang chạy)"
    md.append(f"# Held-out generalization (R3b) — preview [{complete}]\n")
    md.append("> Số liệu 100% tự sinh từ `fs_heldout_results.csv` + stats. KHÔNG gõ tay. "
              "Bảng tự đầy đủ 7 cột khi Stage 2 xong.\n")

    # Ranking + Friedman
    md.append("## Friedman ranking (held-out accuracy, thấp = tốt)\n")
    md.append("| Algo | avg rank |")
    md.append("|---|---|")
    for a, r in s["ranking"].sort_values().items():
        star = " **(ours)**" if a == TARGET else ""
        md.append(f"| {a}{star} | {r:.2f} |")
    if s["stats"]:
        md.append(f"\nFriedman χ²={s['stats']['friedman_chi2']:.3f}, p={s['stats']['friedman_p']:.2e}\n")

    # Wilcoxon summary vs each opponent
    if not s["wil"].empty:
        md.append("## Wilcoxon+Holm vs RG-SCSO (per-dataset paired, α=0.05)\n")
        md.append("| Đối thủ | + (RG thắng) | − (thua) | = (hòa) | median\\|d\\| |")
        md.append("|---|---|---|---|---|")
        for opp in [a for a in algos if a != TARGET]:
            sub = s["wil"][s["wil"].compared_with == opp]
            if sub.empty:
                continue
            plus = (sub["mark"] == "+").sum()
            minus = (sub["mark"] == "-").sum()
            eq = (sub["mark"] == "=").sum()
            md.append(f"| {opp} | {plus} | {minus} | {eq} | {sub['cohens_d'].abs().median():.2f} |")

    # Accuracy table
    md.append("\n## Held-out accuracy (mean ± std, 30 run) — **bold** = best/dataset\n")
    md.append("| Dataset | #F | " + " | ".join(algos) + " |")
    md.append("|---|---|" + "|".join(["---"] * n_algo) + "|")
    for ds in s["datasets"]:
        row = s["acc_mean"].loc[ds]
        best = row.max()
        cells = [ds, str(int(s["ntot"][ds]))]
        for a in algos:
            m, sd = s["acc_mean"].loc[ds, a], s["acc_std"].loc[ds, a]
            txt = f"{m:.4f}±{sd:.3f}"
            cells.append(f"**{txt}**" if abs(m - best) < 1e-9 else txt)
        md.append("| " + " | ".join(cells) + " |")
    mean_acc = s["acc_mean"].mean()
    md.append("| **MEAN** | — | " + " | ".join(f"{mean_acc[a]:.4f}" for a in algos) + " |")

    # #features table
    md.append("\n## Held-out #features selected (mean, 30 run) — **bold** = fewest/dataset\n")
    md.append("| Dataset | " + " | ".join(algos) + " |")
    md.append("|---|" + "|".join(["---"] * n_algo) + "|")
    for ds in s["datasets"]:
        row = s["nf_mean"].loc[ds]
        least = row.min()
        cells = [ds]
        for a in algos:
            v = s["nf_mean"].loc[ds, a]
            cells.append(f"**{v:.1f}**" if abs(v - least) < 1e-9 else f"{v:.1f}")
        md.append("| " + " | ".join(cells) + " |")
    mean_nf = s["nf_mean"].mean()
    md.append("| **MEAN** | " + " | ".join(f"{mean_nf[a]:.1f}" for a in algos) + " |")

    return "\n".join(md) + "\n"


def main() -> None:
    s = load()
    with open(os.path.join(HELDOUT_DIR, "generalization_acc.tex"), "w") as fh:
        fh.write(acc_table_tex(s))
    with open(os.path.join(HELDOUT_DIR, "generalization_nfeat.tex"), "w") as fh:
        fh.write(nfeat_table_tex(s))
    with open("HELDOUT_REVIEW.md", "w") as fh:
        fh.write(markdown_preview(s))
    print(f"Đã sinh: {HELDOUT_DIR}/generalization_acc.tex, generalization_nfeat.tex, HELDOUT_REVIEW.md")
    print(f"Thuật toán trong bảng: {s['algos']}")


if __name__ == "__main__":
    main()
