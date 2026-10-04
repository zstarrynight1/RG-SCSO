"""
make_i5_figures.py — SOP §50 presentation figures generated ONLY from frozen evidence.

No optimizer is re-run. Inputs:
  - experiments/results_fs_signal_position/fs_signal_position_results.csv  (injection ladder V0-V4)
  - experiments/results_stability/stability_masks.csv                      (per-run selected indices)
  - data/processed/<dataset>.csv                                           (for MI relevance axis)

Outputs (figures/):
  1. injection_ladder.pdf          — V0->V4 controlled-ablation contribution
  2. relevance_vs_frequency.pdf     — mechanism: MI relevance vs selection frequency
  3. stability_jaccard.pdf          — run-to-run subset stability (mean pairwise Jaccard)

These make existing evidence easier to inspect; they do not change any scientific result.
"""
from __future__ import annotations

import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

FIG_DIR = "figures"
PROCESSED_DIR = os.path.join("data", "processed")
SIGNAL_CSV = os.path.join("experiments", "results_fs_signal_position", "fs_signal_position_results.csv")
MASKS_CSV = os.path.join("experiments", "results_stability", "stability_masks.csv")

COLORS = {"RG-SCSO": "#d62728", "SCSO": "#1f77b4", "AOA": "#2ca02c"}
MARKERS = {"RG-SCSO": "o", "SCSO": "s", "AOA": "^"}

# V0->V4 ladder: signal-position step name -> (short label, long label)
LADDER = [
    ("1_RandomInit_NoRMS", "V0", "Base SCSO"),
    ("2_MIInit_NoRMS", "V1", "MI init"),
    ("3_MIObjective_NoRMS", "V2", "MI objective"),
    ("4_MITransfer_RMS_NoUMR", "V3", "MI decision (RMS)"),
    ("5_RMS_UMR_Full", "V4", "Full (RMS+UMR)"),
]
DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
DS_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, len(DATASETS)))

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


def _load_xy(name: str):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _parse_mask(s: str) -> set[int]:
    s = str(s).strip()
    if not s or s == "nan":
        return set()
    return {int(t) for t in s.split(";") if t != ""}


# --------------------------------------------------------------------------- #
def fig_injection_ladder() -> None:
    df = pd.read_csv(SIGNAL_CSV)
    df["frac"] = df["n_selected_features"] / df["n_total_features"]
    steps = [s for s, _, _ in LADDER]
    xlabels = [short for _, short, _ in LADDER]
    x = np.arange(len(steps))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for metric, ax, ylab, title in [
        ("accuracy", axes[0], "Mean accuracy", "(a) Accuracy"),
        ("frac", axes[1], "Selected-feature fraction", "(b) Parsimony"),
    ]:
        # per-dataset light lines
        for ds, c in zip(DATASETS, DS_COLORS):
            sub = df[df["dataset"] == ds]
            ys = [sub[sub["step"] == s][metric].mean() for s in steps]
            ax.plot(x, ys, "-", color=c, lw=0.9, alpha=0.55, marker=".", ms=4,
                    label=ds if metric == "accuracy" else None)
        # bold mean-over-datasets line
        ym = [np.mean([df[(df["dataset"] == ds) & (df["step"] == s)][metric].mean()
                       for ds in DATASETS]) for s in steps]
        ax.plot(x, ym, "-", color="black", lw=2.2, marker="o", ms=5, label="Mean" if metric == "accuracy" else None, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel("V0 Base · V1 MI-init · V2 MI-objective · V3 MI-decision(RMS) · V4 Full", fontsize=6.6)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    axes[0].legend(fontsize=6.6, ncol=2, loc="lower right", framealpha=0.9)
    # annotate the decision-stage jump (placed clear of the axis)
    ym3 = np.mean([df[(df.dataset == ds) & (df.step == steps[3])]["accuracy"].mean() for ds in DATASETS])
    axes[0].annotate("gain at\ndecision stage (V3)", xy=(3, ym3), xytext=(1.55, 0.905),
                     fontsize=6.8, color="#444", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    fig.tight_layout()
    _save(fig, "injection_ladder")


# --------------------------------------------------------------------------- #
def _selection_frequency(dataset: str, algo: str, n_total: int) -> np.ndarray:
    df = pd.read_csv(MASKS_CSV)
    sub = df[(df["dataset"] == dataset) & (df["algorithm"] == algo)]
    freq = np.zeros(n_total, dtype=float)
    for s in sub["selected_indices"]:
        for j in _parse_mask(s):
            if 0 <= j < n_total:
                freq[j] += 1.0
    return freq / max(len(sub), 1)


def fig_relevance_vs_frequency() -> None:
    rho_rows = []
    # representative scatter dataset + full bar panel
    scatter_ds = "WDBC"
    fig = plt.figure(figsize=(7.4, 3.2))
    axs = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2)

    for ds in DATASETS:
        X, y = _load_xy(ds)
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        h_y = float(-np.sum(p * np.log(p + 1e-12)))
        mi = mutual_info_classif(X, y, random_state=42)
        rel = np.clip(mi / h_y, 0.0, 1.0) if h_y > 0 else np.full(X.shape[1], 0.5)
        for algo in ("RG-SCSO", "SCSO"):
            freq = _selection_frequency(ds, algo, X.shape[1])
            rho = spearmanr(rel, freq).correlation
            rho_rows.append({"dataset": ds, "algorithm": algo, "spearman_rho": rho})
            if ds == scatter_ds:
                axs.scatter(rel, freq, s=22, alpha=0.75, color=COLORS[algo],
                            marker=MARKERS[algo], edgecolor="white", lw=0.4,
                            label=f"{algo} (ρ={rho:.2f})")

    axs.set_xlabel("MI relevance $\\rho_j$ (full-sample, for visualization)")
    axs.set_ylabel("Selection frequency over 30 runs")
    axs.set_title(f"(a) {scatter_ds}: relevance vs. selection", fontsize=9)
    axs.legend(fontsize=7, loc="upper left")
    axs.grid(ls=":", lw=0.5, alpha=0.6)

    rho_df = pd.DataFrame(rho_rows)
    x = np.arange(len(DATASETS))
    w = 0.38
    for i, algo in enumerate(("RG-SCSO", "SCSO")):
        vals = [rho_df[(rho_df.dataset == ds) & (rho_df.algorithm == algo)]["spearman_rho"].iloc[0] for ds in DATASETS]
        axb.bar(x + (i - 0.5) * w, vals, w, color=COLORS[algo], label=algo, alpha=0.9)
    axb.axhline(0, color="black", lw=0.6)
    axb.set_xticks(x)
    axb.set_xticklabels(DATASETS, rotation=30, ha="right", fontsize=7.5)
    axb.set_ylabel("Spearman $\\rho$(relevance, frequency)")
    axb.set_title("(b) Across datasets", fontsize=9)
    axb.legend(fontsize=7.5)
    axb.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    _save(fig, "relevance_vs_frequency")
    rho_df.to_csv(os.path.join(FIG_DIR, "relevance_vs_frequency_rho.csv"), index=False)
    print("    spearman rho table -> figures/relevance_vs_frequency_rho.csv")


# --------------------------------------------------------------------------- #
def _mean_pairwise_jaccard(masks: list[set[int]]) -> float:
    pairs = list(itertools.combinations(masks, 2))
    if not pairs:
        return float("nan")
    vals = []
    for a, b in pairs:
        u = len(a | b)
        vals.append(len(a & b) / u if u else 1.0)
    return float(np.mean(vals))


STAB_IDX_CSV = os.path.join("experiments", "results_stability", "stability_index_results.csv")


def fig_stability() -> None:
    """Size-corrected Nogueira phi (primary) + subset fraction (explains the raw-Jaccard
    artifact) + raw Jaccard saved to CSV for full transparency."""
    df = pd.read_csv(MASKS_CSV)
    idx = pd.read_csv(STAB_IDX_CSV)
    algos = ["RG-SCSO", "SCSO", "AOA"]
    rows = []
    for ds in DATASETS:
        for algo in algos:
            sub = df[(df.dataset == ds) & (df.algorithm == algo)]
            masks = [_parse_mask(s) for s in sub["selected_indices"]]
            irow = idx[(idx.algorithm == algo) & (idx.dataset == ds)].iloc[0]
            n_tot = float(irow["n_total_features"])
            rows.append({
                "dataset": ds, "algorithm": algo,
                "nogueira_phi": float(irow["nogueira_phi"]),
                "mean_selected_fraction": float(irow["mean_n_selected"]) / n_tot,
                "raw_mean_jaccard": _mean_pairwise_jaccard(masks),
            })
    sdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    x = np.arange(len(DATASETS))
    w = 0.26
    # (a) size-/chance-corrected Nogueira phi
    for i, algo in enumerate(algos):
        vals = [sdf[(sdf.dataset == ds) & (sdf.algorithm == algo)]["nogueira_phi"].iloc[0] for ds in DATASETS]
        axes[0].bar(x + (i - 1) * w, vals, w, color=COLORS[algo], label=algo, alpha=0.9)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Nogueira $\\hat{\\Phi}$ (size-/chance-corrected)")
    axes[0].set_title("(a) Corrected stability", fontsize=9)
    axes[0].legend(fontsize=7.2); axes[0].grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    # (b) mean selected fraction -> explains why raw Jaccard misleads
    for i, algo in enumerate(algos):
        vals = [sdf[(sdf.dataset == ds) & (sdf.algorithm == algo)]["mean_selected_fraction"].iloc[0] for ds in DATASETS]
        axes[1].bar(x + (i - 1) * w, vals, w, color=COLORS[algo], label=algo, alpha=0.9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Mean selected fraction")
    axes[1].set_title("(b) Subset size (AOA selects ~all)", fontsize=9)
    axes[1].legend(fontsize=7.2); axes[1].grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    _save(fig, "stability_metrics")
    sdf.to_csv(os.path.join(FIG_DIR, "stability_metrics.csv"), index=False)
    print("    stability table (phi + fraction + raw Jaccard) -> figures/stability_metrics.csv")


if __name__ == "__main__":
    print("I5 figures (from frozen evidence only):")
    fig_injection_ladder()
    fig_relevance_vs_frequency()
    fig_stability()
    print("done.")
