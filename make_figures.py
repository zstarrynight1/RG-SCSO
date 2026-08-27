"""Phase R5 — sinh hình cho bài RG-SCSO (IEEE Transactions).

Bốn hình, tất cả lưu CẢ .pdf (vector, cho LaTeX) VÀ .png (300 dpi) vào figures/:
    1. concept.pdf      — schematic washout vs RG-SCSO (vẽ tay bằng matplotlib).
    2. cd_diagram.pdf   — Critical-Difference (Friedman + Nemenyi) trên 18 dataset.
    3. convergence.pdf  — mean best fitness theo iteration (low-dim + high-dim).
    4. mechanism.pdf    — overlap feature RG-SCSO chọn với top-MI vs baseline.

Hình 1-2 chỉ cần artifact CÓ SẴN (fs_results.csv / friedman_ranking.csv) → chạy
được ngay. Hình 3-4 cần convergence-curve + mask KHÔNG lưu trong fs_results.csv,
nên đọc từ figures/fig_capture.npz do `capture_fig_data.py` tái tạo (deterministic,
seed 42+run_id — chỉ để VẼ, không đổi số báo cáo). Thiếu file → bỏ qua + cảnh báo.

NGUYÊN TẮC: số trên hình sinh tự động từ artifact, KHÔNG hardcode. Màu 1 thuật
toán = 1 màu cố định ở mọi hình (đối chiếu chéo dễ khi reviewer soi).

Chạy: .venv/bin/python make_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless, CPU-only, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "figures"
FS_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
RANK_CSV = os.path.join("experiments", "results_fs", "friedman_ranking.csv")
CAPTURE_NPZ = os.path.join(FIG_DIR, "fig_capture.npz")

# Bộ 7 thuật toán khóa (pre-registered) + năm để chú thích.
LOCKED = ["RG-SCSO", "SCSO", "AOA", "COA", "GWO", "PSO", "RIME"]
YEAR = {"RG-SCSO": "ours", "SCSO": "2022", "AOA": "2021", "COA": "2023",
        "GWO": "2014", "PSO": "1995", "RIME": "2023"}

# Màu cố định: RG-SCSO đỏ nổi bật, SCSO xanh (baseline gốc), còn lại phân biệt.
COLORS = {
    "RG-SCSO": "#d62728", "SCSO": "#1f77b4", "AOA": "#2ca02c",
    "COA": "#9467bd", "GWO": "#8c564b", "PSO": "#e377c2", "RIME": "#ff7f0e",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  ✓ figures/{name}.pdf + .png")


# ----------------------------------------------------------------- 1. concept
def fig_concept() -> None:
    """Schematic 2 tầng: pipeline thường (washout) vs RG-SCSO (per-feature R)."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 4.2))

    def box(ax, x, w, label, fc, tc="black"):
        ax.add_patch(plt.Rectangle((x, 0.30), w, 0.40, facecolor=fc,
                                    edgecolor="black", lw=1.0))
        ax.text(x + w / 2, 0.50, label, ha="center", va="center",
                fontsize=8.5, color=tc, wrap=True)

    def arrow(ax, x0, x1):
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))

    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")

    # --- Top: conventional pipeline ---
    ax_top.text(0.0, 0.92, "(a) Conventional continuous-to-binary pipeline",
                fontsize=9, fontweight="bold")
    box(ax_top, 0.2, 2.0, "Continuous\nsearch operator", "#eaf2fb")
    arrow(ax_top, 2.3, 2.9)
    box(ax_top, 2.9, 2.4, "Fixed, feature-agnostic\ntransfer (sigmoid)", "#fde8e8")
    arrow(ax_top, 5.4, 6.0)
    box(ax_top, 6.0, 1.8, "Binary mask", "#eaf2fb")
    ax_top.text(4.1, 0.16, "quantization discards fine adjustments  →  WASHOUT",
                ha="center", fontsize=8, color="#b22222", style="italic")

    # --- Bottom: RG-SCSO ---
    ax_bot.text(0.0, 0.92, "(b) RG-SCSO: sensitivity range → per-feature bit-flip",
                fontsize=9, fontweight="bold")
    box(ax_bot, 0.2, 2.0, "SCSO position\nupdate  (range R)", "#eaf2fb")
    arrow(ax_bot, 2.3, 2.9)
    box(ax_bot, 2.9, 2.6, "RMS: relevance-\nmodulated V-transfer", "#e8f6ea", )
    arrow(ax_bot, 5.6, 6.2)
    box(ax_bot, 6.2, 1.8, "Binary mask", "#eaf2fb")
    # relevance field feeding RMS + UMR
    ax_bot.add_patch(plt.Rectangle((2.9, 0.72), 2.6, 0.16, facecolor="#fff4d6",
                                   edgecolor="black", lw=0.8))
    ax_bot.text(4.2, 0.80, "ρ: mutual-information relevance field", ha="center",
                va="center", fontsize=7.5)
    ax_bot.annotate("", xy=(4.2, 0.71), xytext=(4.2, 0.70),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#c8a02c"))
    ax_bot.text(8.0, 0.24, "UMR: memetic refinement on uncertain bits",
                ha="center", fontsize=7.5, color="#333333", style="italic")

    fig.suptitle("")
    fig.tight_layout()
    _save(fig, "concept")


# ---------------------------------------------------------------- 2. CD diagram
def _nemenyi_q(k: int) -> float:
    """Giá trị tới hạn Nemenyi q_alpha (alpha=0.05) theo số thuật toán k."""
    table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}
    return table[k]


def fig_cd() -> None:
    """Critical-Difference diagram (Demšar 2006): trục rank, thanh nối các nhóm
    KHÔNG khác biệt có ý nghĩa (chênh rank < CD)."""
    df = pd.read_csv(FS_CSV)
    df = df[df["algorithm"].isin(LOCKED)]
    piv = df.groupby(["dataset", "algorithm"])["accuracy"].mean().unstack()[LOCKED]
    ranks = piv.rank(axis=1, ascending=False)  # rank 1 = accuracy cao nhất
    avg = ranks.mean().sort_values()           # thứ tự tốt→tệ
    k, N = piv.shape[1], piv.shape[0]
    CD = _nemenyi_q(k) * np.sqrt(k * (k + 1) / (6.0 * N))

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # trục rank (1 tốt nhất bên trái)
    axis_y = 0.78
    ax.plot([lo, hi], [axis_y, axis_y], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y, axis_y + 0.03], "k-", lw=1.0)
        ax.text(r, axis_y + 0.06, str(r), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, axis_y + 0.15, "Average rank (1 = best)",
            ha="center", fontsize=8.5)

    names = list(avg.index)
    half = (len(names) + 1) // 2
    # nửa tốt hơn (rank nhỏ) gắn nhãn bên trái, nửa còn lại bên phải
    for i, a in enumerate(names):
        r = avg[a]
        if i < half:  # trái
            xtext, yr = lo - 0.4, axis_y - 0.10 - 0.11 * i
        else:         # phải
            xtext, yr = hi + 0.4, axis_y - 0.10 - 0.11 * (len(names) - 1 - i)
        ha = "right" if i < half else "left"
        ax.plot([r, r], [axis_y, yr], color=COLORS[a], lw=1.1)
        ax.plot([r, xtext], [yr, yr], color=COLORS[a], lw=1.1)
        lbl = f"{a} ({YEAR[a]})  {r:.2f}"
        ax.text(xtext + (-0.05 if ha == "right" else 0.05), yr, lbl,
                ha=ha, va="center", fontsize=8,
                fontweight="bold" if a == "RG-SCSO" else "normal",
                color=COLORS[a] if a == "RG-SCSO" else "black")

    # thanh CD: nhóm các thuật toán liên tiếp có chênh rank <= CD
    bar_y = axis_y - 0.05
    groups = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and (avg[names[j + 1]] - avg[names[i]]) <= CD:
            j += 1
        if j > i:
            groups.append((avg[names[i]], avg[names[j]]))
    # loại nhóm con bị bao
    groups = [g for g in groups if not any(g2[0] <= g[0] and g[1] <= g2[1]
                                           and g2 != g for g2 in groups)]
    for gi, (r0, r1) in enumerate(groups):
        yy = bar_y - 0.045 * gi
        ax.plot([r0 - 0.03, r1 + 0.03], [yy, yy], "k-", lw=2.6, solid_capstyle="round")

    # thước CD ở góc trên trái
    ax.plot([lo, lo + CD], [0.96, 0.96], "k-", lw=1.6)
    ax.plot([lo, lo], [0.945, 0.975], "k-", lw=1.0)
    ax.plot([lo + CD, lo + CD], [0.945, 0.975], "k-", lw=1.0)
    ax.text(lo + CD / 2, 0.99, f"CD = {CD:.2f}", ha="center", va="bottom",
            fontsize=8)
    fig.tight_layout()
    _save(fig, "cd_diagram")
    print(f"    (Nemenyi CD={CD:.3f}, k={k}, N={N}; groups joined = not sig. diff.)")


# ----------------------------------------------------------- 2b. cd, held-out
HELDOUT_RANK_CSV = os.path.join("experiments", "results_fs_heldout", "friedman_ranking.csv")


def fig_cd_heldout() -> None:
    """Critical-Difference diagram trên RANKING HELD-OUT (leak-free), thay cho
    bản in-sample làm bằng chứng chính ở main text (RG-SCSO_MASTER_FINAL_
    COMPLETE.md Section 15 / Section 32 item 4: "Held-out phải là evidence
    chính; main CD diagram phải ưu tiên held-out"). Cùng công thức Nemenyi,
    cùng style, chỉ khác nguồn rank (friedman_ranking.csv của held-out thay
    vì in-sample)."""
    if not os.path.exists(HELDOUT_RANK_CSV):
        print("  ⚠ friedman_ranking.csv (held-out) chưa có → bỏ qua cd_diagram_heldout.")
        return
    rank = pd.read_csv(HELDOUT_RANK_CSV).set_index("algorithm")["avg_rank"]
    rank = rank[[a for a in LOCKED if a in rank.index]]
    avg = rank.sort_values()
    k = len(rank)
    n_ds = pd.read_csv(os.path.join("experiments", "results_fs_heldout",
                                     "fs_heldout_results.csv"))["dataset"].nunique()
    N = n_ds
    CD = _nemenyi_q(k) * np.sqrt(k * (k + 1) / (6.0 * N))

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    axis_y = 0.78
    ax.plot([lo, hi], [axis_y, axis_y], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y, axis_y + 0.03], "k-", lw=1.0)
        ax.text(r, axis_y + 0.06, str(r), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, axis_y + 0.15, "Average held-out rank (1 = best)",
            ha="center", fontsize=8.5)

    names = list(avg.index)
    half = (len(names) + 1) // 2
    for i, a in enumerate(names):
        r = avg[a]
        if i < half:
            xtext, yr = lo - 0.4, axis_y - 0.10 - 0.11 * i
        else:
            xtext, yr = hi + 0.4, axis_y - 0.10 - 0.11 * (len(names) - 1 - i)
        ha = "right" if i < half else "left"
        ax.plot([r, r], [axis_y, yr], color=COLORS[a], lw=1.1)
        ax.plot([r, xtext], [yr, yr], color=COLORS[a], lw=1.1)
        lbl = f"{a} ({YEAR[a]})  {r:.2f}"
        ax.text(xtext + (-0.05 if ha == "right" else 0.05), yr, lbl,
                ha=ha, va="center", fontsize=8,
                fontweight="bold" if a == "RG-SCSO" else "normal",
                color=COLORS[a] if a == "RG-SCSO" else "black")

    bar_y = axis_y - 0.05
    groups = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and (avg[names[j + 1]] - avg[names[i]]) <= CD:
            j += 1
        if j > i:
            groups.append((avg[names[i]], avg[names[j]]))
    groups = [g for g in groups if not any(g2[0] <= g[0] and g[1] <= g2[1]
                                           and g2 != g for g2 in groups)]
    for gi, (r0, r1) in enumerate(groups):
        yy = bar_y - 0.045 * gi
        ax.plot([r0 - 0.03, r1 + 0.03], [yy, yy], "k-", lw=2.6, solid_capstyle="round")

    ax.plot([lo, lo + CD], [0.96, 0.96], "k-", lw=1.6)
    ax.plot([lo, lo], [0.945, 0.975], "k-", lw=1.0)
    ax.plot([lo + CD, lo + CD], [0.945, 0.975], "k-", lw=1.0)
    ax.text(lo + CD / 2, 0.99, f"CD = {CD:.2f}", ha="center", va="bottom",
            fontsize=8)
    fig.tight_layout()
    _save(fig, "cd_diagram_heldout")
    print(f"    (held-out Nemenyi CD={CD:.3f}, k={k}, N={N}; "
          f"groups joined = not sig. diff.)")


# --------------------------------------------------------------- 3. convergence
def fig_convergence() -> None:
    if not os.path.exists(CAPTURE_NPZ):
        print("  ⚠ figures/fig_capture.npz chưa có → bỏ qua convergence "
              "(chạy capture_fig_data.py trước).")
        return
    d = np.load(CAPTURE_NPZ, allow_pickle=True)
    curves = d["curves"].item()          # curves[dataset][algo] = (n_runs, T)
    conv_ds = list(d["conv_datasets"])
    n_runs = int(d["n_runs"])

    fig, axes = plt.subplots(1, len(conv_ds), figsize=(7.0, 3.0), squeeze=False)
    for ci, ds in enumerate(conv_ds):
        ax = axes[0][ci]
        for a in LOCKED:
            if ds in curves and a in curves[ds]:
                arr = np.asarray(curves[ds][a])         # (runs, T)
                mean = arr.mean(axis=0)
                it = np.arange(1, len(mean) + 1)
                ax.plot(it, mean, color=COLORS[a], lw=1.4 if a == "RG-SCSO" else 1.0,
                        label=f"{a}", zorder=3 if a == "RG-SCSO" else 2)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8.5)
        if ci == 0:
            ax.set_ylabel("Mean best fitness", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0][0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(f"Convergence (mean over {n_runs} runs)", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "convergence")


# --------------------------------- 4b. convergence, feature-selection objective
CONVERGENCE_FS_CSV = os.path.join(
    "experiments", "results_convergence", "convergence_curves.csv"
)


def fig_convergence_fs() -> None:
    """Convergence trên CHÍNH objective feature-selection (khác fig_convergence()
    ở trên, vốn dùng dữ liệu capture cũ từ benchmark liên tục). Nguồn:
    run_convergence_curves.py — RG-SCSO/SCSO/AOA, 3 dataset (Zoo/WDBC/ColonCancer),
    5 run, minh họa hành vi tìm kiếm, KHÔNG phải một claim thống kê mới."""
    if not os.path.exists(CONVERGENCE_FS_CSV):
        print("  ⚠ convergence_curves.csv chưa có → bỏ qua fig_convergence_fs "
              "(chạy run_convergence_curves.py trước).")
        return
    df = pd.read_csv(CONVERGENCE_FS_CSV)
    datasets = ["Zoo", "WDBC", "ColonCancer"]
    algos = ["RG-SCSO", "SCSO", "AOA"]
    n_runs = df["run_id"].nunique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0, 3.0), squeeze=False)
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        for a in algos:
            sub = df[(df.dataset == ds) & (df.algorithm == a)]
            mean = sub.groupby("iteration")["fitness"].mean()
            ax.plot(mean.index, mean.values, color=COLORS[a],
                     lw=1.4 if a == "RG-SCSO" else 1.0, label=a,
                     zorder=3 if a == "RG-SCSO" else 2)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8.5)
        if ci == 0:
            ax.set_ylabel("Mean best fitness", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0][0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(f"Convergence on the feature-selection objective "
                 f"(mean over {n_runs} runs)", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "convergence_fs")


# --------------------------------------- 4b. accuracy-parsimony trade-off
def fig_accuracy_parsimony_tradeoff() -> None:
    """Held-out accuracy vs. mean selected-feature FRACTION (nfeat/ntot,
    normalized so datasets with 8 vs. 3571 features are comparable), one
    point per algorithm, averaged across all 18 datasets. RG-SCSO_MASTER_
    FINAL_COMPLETE.md Section 24, Figure 4 -- plotted entirely from existing
    held-out data (build_heldout_table.load()), no new experiment."""
    import build_heldout_table as _heldout

    hs = _heldout.load()
    acc_mean, nf_mean, ntot = hs["acc_mean"], hs["nf_mean"], hs["ntot"]
    frac = nf_mean.div(ntot, axis=0)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    pts = {a: (float(frac[a].mean()), float(acc_mean[a].mean())) for a in hs["algos"]}
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span, y_span = x_max - x_min, y_max - y_min

    # Generous, even padding on every side (not just the side a label
    # happens to overflow into) -- a fixed points-offset label near any
    # edge of the data range would otherwise sit outside the axes frame.
    x_pad, y_pad = x_span * 0.16, y_span * 0.14
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for a, (x, y) in pts.items():
        ax.scatter(x, y, s=60 if a == "RG-SCSO" else 46,
                   color=COLORS[a], zorder=3 if a == "RG-SCSO" else 2,
                   edgecolors="black", linewidths=0.6)
        # Flip the label to whichever side of the marker has more room
        # left before the axes edge, so a point near the left or right
        # boundary (e.g. AOA at the far right) never has its text pushed
        # past the frame.
        near_right = (x_max + x_pad - x) < 0.22 * (x_span + 2 * x_pad)
        near_left = (x - (x_min - x_pad)) < 0.14 * (x_span + 2 * x_pad)
        if near_right and not near_left:
            dx, ha = -6, "right"
        else:
            dx, ha = 6, "left"
        near_top = (y_max + y_pad - y) < 0.12 * (y_span + 2 * y_pad)
        dy, va = (-8, "top") if near_top else (4, "bottom")
        ax.annotate(a, (x, y), fontsize=7.5, xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va)
    ax.set_xlabel("Mean selected-feature fraction (lower = sparser)", fontsize=8.5)
    ax.set_ylabel("Mean held-out accuracy", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    _save(fig, "accuracy_parsimony_tradeoff")


# ------------------------------------------------- 4c. threshold heatmap
THRESHOLD_CSV = os.path.join("experiments", "results_threshold",
                              "threshold_sensitivity_results.csv")


def fig_threshold_heatmap() -> None:
    """Heatmap accuracy theo (dataset x tau), số feature trung bình ghi kèm
    trong ô (cùng tinh thần "accuracy (features)" đã dùng ở Table 1/4/5) --
    trực quan hóa dữ liệu threshold-sensitivity đã có ở Supplementary Table
    (RG-SCSO_MASTER_FINAL_COMPLETE.md Section 24, Figure 6)."""
    if not os.path.exists(THRESHOLD_CSV):
        print("  ⚠ threshold_sensitivity_results.csv chưa có → bỏ qua threshold_heatmap.")
        return
    df = pd.read_csv(THRESHOLD_CSV)
    datasets = sorted(df["dataset"].unique())
    taus = sorted(df["tau"].unique())
    acc = df.pivot(index="dataset", columns="tau", values="mean_accuracy").loc[datasets, taus]
    nfeat = df.pivot(index="dataset", columns="tau", values="mean_n_selected").loc[datasets, taus]

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(acc.to_numpy(), cmap="RdYlGn", aspect="auto",
                    vmin=acc.to_numpy().min() - 0.01, vmax=acc.to_numpy().max() + 0.01)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"$\\tau$={t:.1f}" for t in taus], fontsize=8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8)
    for i in range(len(datasets)):
        for j in range(len(taus)):
            a = acc.iloc[i, j]
            nf = nfeat.iloc[i, j]
            ax.text(j, i, f"{a:.3f}\n({nf:.0f})", ha="center", va="center",
                    fontsize=6.6, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Mean accuracy", fontsize=8)
    ax.set_title("Threshold sensitivity: accuracy (mean #features)", fontsize=9)
    fig.tight_layout()
    _save(fig, "threshold_heatmap")


# ------------------------------------------------------ 4d. graphical abstract
def fig_graphical_abstract() -> None:
    """Elsevier Graphical Abstract for the ASOC-target manuscript (RG-SCSO_
    MASTER_FINAL_COMPLETE.md §35b): a single portrait top-to-bottom flow,
    reusing fig_concept()'s box()/arrow() drawing language but restructured
    from two side-by-side panels into one vertical pipeline, since a GA is
    read as one image, not a two-part comparison. Raster export is sized to
    land at exactly 531x1328 px (dpi=200, figsize=531/200 x 1328/200 in),
    the pixel spec quoted in the master plan; also saves a vector PDF."""
    dpi = 200
    fig, ax = plt.subplots(figsize=(531 / dpi, 1328 / dpi))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 27.2)
    ax.axis("off")

    def box(y, h, label, fc, tc="black", fontsize=7.2):
        ax.add_patch(plt.Rectangle((0.6, y), 8.8, h, facecolor=fc,
                                    edgecolor="black", lw=1.0))
        ax.text(5.0, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=tc, wrap=True)

    def down_arrow(y0, y1):
        ax.annotate("", xy=(5.0, y1), xytext=(5.0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))

    box(23.4, 2.0, "Dataset + per-feature\nmutual-information relevance prior",
        "#eaf2fb")
    down_arrow(23.4, 21.3)
    box(19.0, 2.0, "SCSO continuous search\n(sensitivity range $R(t)$)", "#eaf2fb")
    down_arrow(19.0, 16.9)
    box(14.4, 2.4, "Relevance-modulated binarization\n(novelty: bias injected"
                   " at the transfer output)", "#fff4d6", tc="#7a5c00",
        fontsize=7.4)
    down_arrow(14.4, 12.3)
    box(10.0, 2.0, "Memetic refinement on\nuncertain bits (UMR)", "#e8f6ea")
    down_arrow(10.0, 7.9)
    box(5.4, 2.2, "Parsimonious binary\nfeature subset", "#eaf2fb")
    down_arrow(5.4, 3.3)
    box(1.0, 2.0, "Competitive held-out\nclassification accuracy", "#eaf2fb")

    ax.text(5.0, 26.5, "RG-SCSO: relevance-guided binarization for\n"
                       "parsimonious feature selection",
            ha="center", va="center", fontsize=8.0, fontweight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    os.makedirs(FIG_DIR, exist_ok=True)
    # Fixed pixel spec (531x1328) requires the untrimmed canvas -- the
    # module-wide savefig.bbox="tight" rcParam would recompute the bbox from
    # content extents and drift off-spec, so it's disabled for this export.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(os.path.join(FIG_DIR, "graphical_abstract.pdf"))
        fig.savefig(os.path.join(FIG_DIR, "graphical_abstract.png"), dpi=dpi)
        fig.savefig(os.path.join(FIG_DIR, "graphical_abstract.tiff"), dpi=dpi)
    plt.close(fig)
    print("  ✓ figures/graphical_abstract.pdf + .png + .tiff (531x1328 px)")


# ------------------------------------------------ 5. diversity / bit-freezing
DIVERSITY_CSV = os.path.join("experiments", "results_diversity", "diversity_history.csv")
GAMMA_COLORS = {0.0: "#1f77b4", 0.5: "#d62728", 1.0: "#2ca02c"}


def fig_diversity() -> None:
    """Đa dạng quần thể (Hamming kỳ vọng) + tỉ lệ bit đóng băng theo vòng lặp,
    ở γ ∈ {0, 0.5, 1.0} — bằng chứng thực nghiệm cho §1.1/§1.2 (Diem_yeu_RG-SCSO.md):
    kiểm tra RMS có làm sụp đa dạng / đóng băng bit sớm hơn V-shaped thuần không."""
    if not os.path.exists(DIVERSITY_CSV):
        print("  ⚠ diversity_history.csv chưa có → bỏ qua fig_diversity "
              "(chạy measure_diversity.py trước).")
        return
    df = pd.read_csv(DIVERSITY_CSV)
    datasets = list(df["dataset"].unique())
    gammas = sorted(df["gamma"].unique())

    fig, axes = plt.subplots(2, len(datasets), figsize=(7.0, 4.4), squeeze=False)
    for ci, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        for g in gammas:
            s = sub[sub.gamma == g].sort_values("iter")
            axes[0][ci].plot(s["iter"], s["diversity"], color=GAMMA_COLORS[g],
                             lw=1.3, label=f"$\\gamma$={g:.1f}")
            axes[1][ci].plot(s["iter"], s["frozen_frac"], color=GAMMA_COLORS[g], lw=1.3)
        axes[0][ci].set_title(ds, fontsize=9)
        axes[1][ci].set_xlabel("Iteration", fontsize=8.5)
        for row in (0, 1):
            axes[row][ci].tick_params(labelsize=7.5)
            axes[row][ci].grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0][0].set_ylabel("Population diversity", fontsize=8.5)
    axes[1][0].set_ylabel("Frozen-bit fraction", fontsize=8.5)
    axes[0][0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "diversity")


# ----------------------------------------------------------------- 4. mechanism
STABILITY_CSV = os.path.join("experiments", "results_stability", "stability_index_results.csv")


def fig_mechanism() -> None:
    """2-panel mechanism-evidence figure (user-requested denser main-text
    figure, merging two comparisons into one display-item to stay within
    Scientific Reports' 8-item cap while showing more evidence).

    (a) Enrichment của tập feature được chọn trong top-MI, chuẩn hóa theo
    CHANCE. precision@|S| thô lệ thuộc |S| (chọn nhiều → chance cao); chance
    của phép chọn ngẫu nhiên |S| feature so với top-|S| MI chính là |S|/N. Ta
    vẽ LIFT = precision ÷ (|S|/N): =1 nghĩa ngẫu nhiên, >1 nghĩa giàu feature
    liên quan. |S| lấy từ fs_results.csv (per-run, cùng seed → khớp
    deterministic).

    (b) Nogueira stability index (Phi) qua RG-SCSO/SCSO/AOA trên 5 dataset
    đại diện, từ experiments/results_stability/stability_index_results.csv
    (Diem_yeu_RG-SCSO.md §2.2) — trực quan hóa phát hiện: RG-SCSO ổn định
    hơn SCSO cùng họ trên mọi dataset, rõ nhất ở 3 dataset chiều thấp, nhưng
    gần mức ngẫu nhiên trên cả 2 dataset gene-expression.
    """
    if not os.path.exists(CAPTURE_NPZ):
        print("  ⚠ figures/fig_capture.npz chưa có → bỏ qua mechanism.")
        return
    d = np.load(CAPTURE_NPZ, allow_pickle=True)
    overlap = d["overlap"].item()        # overlap[dataset][algo] = (n_runs,) precision@|S|
    mech_ds = list(d["mech_datasets"])
    n_runs = int(d["n_runs"])
    algos = [a for a in ["RG-SCSO", "SCSO"] if any(a in overlap.get(ds, {}) for ds in mech_ds)]

    df = pd.read_csv(FS_CSV)

    def _lift(ds: str, a: str) -> np.ndarray:
        """precision ÷ chance per-run; |S| khớp theo run_id đã sort."""
        sub = df[(df.dataset == ds) & (df.algorithm == a)].sort_values("run_id")
        n_total = int(sub.n_total_features.iloc[0])
        nsel = sub.n_selected_features.to_numpy(dtype=float)
        prec = np.asarray(overlap[ds][a], dtype=float)
        chance = nsel[: len(prec)] / n_total
        return prec / chance

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 3.0))

    x = np.arange(len(mech_ds))
    w = 0.36
    for ai, a in enumerate(algos):
        means = [float(np.mean(_lift(ds, a))) for ds in mech_ds]
        stds = [float(np.std(_lift(ds, a))) for ds in mech_ds]
        ax_a.bar(x + (ai - 0.5) * w, means, w, yerr=stds, capsize=3,
                 color=COLORS[a], label=a, edgecolor="black", lw=0.6, zorder=3)
    ax_a.axhline(1.0, color="black", ls="--", lw=1.0, zorder=2,
                 label="chance (relevance-agnostic)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(mech_ds, fontsize=8.5)
    ax_a.set_ylabel("Top-MI enrichment  (precision ÷ chance)", fontsize=8.5)
    ax_a.set_ylim(0.9, max(1.35, ax_a.get_ylim()[1]))
    ax_a.tick_params(labelsize=8)
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax_a.set_title(f"(a) Relevance-guided enrichment (mean±std, {n_runs} runs)",
                   loc="left", fontsize=9)

    if os.path.exists(STABILITY_CSV):
        st = pd.read_csv(STABILITY_CSV)
        stab_ds = sorted(st["dataset"].unique())
        stab_algos = ["RG-SCSO", "SCSO", "AOA"]
        xb = np.arange(len(stab_ds))
        wb = 0.26
        for ai, a in enumerate(stab_algos):
            vals = [float(st[(st.dataset == ds) & (st.algorithm == a)]["nogueira_phi"].iloc[0])
                    for ds in stab_ds]
            ax_b.bar(xb + (ai - 1) * wb, vals, wb, color=COLORS.get(a, "#888888"),
                      label=a, edgecolor="black", lw=0.6, zorder=3)
        ax_b.axhline(0.0, color="black", lw=0.8, zorder=2)
        ax_b.set_xticks(xb)
        ax_b.set_xticklabels(stab_ds, fontsize=8, rotation=20, ha="right")
        ax_b.set_ylabel(r"Stability ($\Phi$)", fontsize=8.5)
        ax_b.tick_params(labelsize=8)
        ax_b.legend(fontsize=7, loc="upper right")
        ax_b.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
        ax_b.set_title("(b) Feature-selection stability (30 runs)", loc="left", fontsize=9)
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "stability data not available", ha="center", va="center")
        print("  ⚠ stability_index_results.csv chưa có → panel (b) trống.")

    fig.tight_layout()
    _save(fig, "mechanism")


def main() -> None:
    print("Sinh hình RG-SCSO →", FIG_DIR)
    fig_concept()
    fig_cd()
    fig_cd_heldout()
    fig_convergence()
    fig_convergence_fs()
    fig_accuracy_parsimony_tradeoff()
    fig_threshold_heatmap()
    fig_mechanism()
    fig_diversity()
    fig_graphical_abstract()
    print("Xong.")


if __name__ == "__main__":
    main()
