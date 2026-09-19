
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

FIGURES_DIR = os.path.join("experiments", "figures")

ALGORITHMS_ORDER = [
    "ECL-SCSO",
    "SCSO",
    "PSO",
    "GWO",
    "HHO",
    "WOA",
    "COA",
    "SSA",
    "OOA",
    "AOA",
    "GA",
]
_OTHER_PALETTE = sns.color_palette("husl", n_colors=len(ALGORITHMS_ORDER) - 2)
ALGORITHM_COLORS = {
    "ECL-SCSO": "#d62728",                                   
    "SCSO": "#1f77b4",                                                    
    **dict(zip(ALGORITHMS_ORDER[2:], _OTHER_PALETTE)),
}

ABLATION_CONFIGS_ORDER = [
    "Full",
    "OnlyDEMutation",
    "OnlyLevyFlight",
    "OnlyAdaptiveR",
    "OnlyChaoticInit",
    "NoImprovement",
]
_ABLATION_PALETTE = sns.color_palette("flare", n_colors=len(ABLATION_CONFIGS_ORDER) - 1)
ABLATION_COLORS = {
    "Full": "#d62728",
    **dict(zip(ABLATION_CONFIGS_ORDER[1:-1], _ABLATION_PALETTE)),
    "NoImprovement": ALGORITHM_COLORS["SCSO"],                                               
}


def _save_figure(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence_curve(
    curves: dict[str, list[float]], title: str, name: str, log_scale: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for algo, curve in curves.items():
        color = ALGORITHM_COLORS.get(algo, "#888888")
        linewidth = 2.4 if algo == "ECL-SCSO" else 1.6
        linestyle = "-" if algo in ("ECL-SCSO", "SCSO") else "--"
        ax.plot(curve, label=algo, color=color, linewidth=linewidth, linestyle=linestyle)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (log scale)" if log_scale else "Best fitness")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    _save_figure(fig, name)


def plot_boxplot_distribution(
    df: pd.DataFrame, value_col: str, group_label: str, name: str
) -> None:
    algos_present = [a for a in ALGORITHMS_ORDER if a in df["algorithm"].unique()]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.boxplot(
        data=df,
        x="algorithm",
        y=value_col,
        order=algos_present,
        hue="algorithm",
        palette=ALGORITHM_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(value_col)
    ax.set_title(f"Phân phối {value_col} qua 30 run — {group_label}")
    ax.tick_params(axis="x", rotation=35)
    _save_figure(fig, name)


def plot_ranking_heatmap(rank_pivot: pd.DataFrame, name: str) -> None:
    cols = [a for a in ALGORITHMS_ORDER if a in rank_pivot.columns]
    rank_pivot = rank_pivot[cols]
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.35 * len(rank_pivot) + 2))
    sns.heatmap(
        rank_pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Rank (1 = tốt nhất)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Ranking heatmap (Friedman) — algorithm x dataset")
    _save_figure(fig, name)


def plot_selected_features_bar(df: pd.DataFrame, name: str) -> None:
    agg = df.groupby(["dataset", "algorithm"], as_index=False).agg(
        ratio=("n_selected_features", "mean"), n_total=("n_total_features", "first")
    )
    agg["ratio"] = agg["ratio"] / agg["n_total"]
    algos_present = [a for a in ALGORITHMS_ORDER if a in agg["algorithm"].unique()]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * agg["dataset"].nunique()), 5))
    sns.barplot(
        data=agg,
        x="dataset",
        y="ratio",
        hue="algorithm",
        hue_order=algos_present,
        palette=ALGORITHM_COLORS,
        ax=ax,
    )
    ax.set_ylabel("Tỉ lệ feature được chọn (selected / total)")
    ax.set_xlabel("")
    ax.set_title("Tỉ lệ feature được chọn trung bình — theo thuật toán, theo dataset")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save_figure(fig, name)


def plot_ablation_bar(ablation_summary: pd.DataFrame, name: str) -> None:
    agg = ablation_summary.groupby("config_name", as_index=False)["mean"].mean()
    agg["config_name"] = pd.Categorical(
        agg["config_name"], categories=ABLATION_CONFIGS_ORDER, ordered=True
    )
    agg = agg.sort_values("config_name")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=agg,
        x="config_name",
        y="mean",
        hue="config_name",
        palette=ABLATION_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean best fitness qua các hàm (log scale)")
    ax.set_xlabel("")
    ax.set_title("Ablation study — đóng góp từng cải tiến của ECL-SCSO")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, name)


def _generate_benchmark_convergence_figures() -> None:
    from config import MAX_ITERATION, POPULATION_SIZE, RANDOM_SEED_BASE
    from src.algorithms.baselines import run_mealpy_baseline
    from src.algorithms.ecl_scso import ECLSCSO
    from src.algorithms.scso import SCSO
    from src.benchmark.cec_functions import build_function, get_function_names

    ranking_path = os.path.join("experiments", "results_benchmark", "friedman_ranking.csv")
    ranking = pd.read_csv(ranking_path).set_index("algorithm")["avg_rank"]
    top_baselines = [a for a in ranking.sort_values().index if a not in ("ECL-SCSO",)][:4]
    algorithms = ["ECL-SCSO"] + top_baselines
    print(f"Convergence curves dùng: {algorithms}")

    for fname in get_function_names(dim=30):
        bf = build_function(fname, dim=30)
        curves = {}
        for algo in algorithms:
            seed = RANDOM_SEED_BASE
            if algo == "SCSO":
                result = SCSO(
                    bf.obj_func, bf.dim, bf.lb, bf.ub, POPULATION_SIZE, MAX_ITERATION, seed
                ).optimize()
            elif algo == "ECL-SCSO":
                result = ECLSCSO(
                    bf.obj_func, bf.dim, bf.lb, bf.ub, POPULATION_SIZE, MAX_ITERATION, seed
                ).optimize()
            else:
                result = run_mealpy_baseline(
                    algo, bf.obj_func, bf.dim, bf.lb, bf.ub, POPULATION_SIZE, MAX_ITERATION, seed
                )
            curves[algo] = result["convergence_curve"]
        plot_convergence_curve(curves, title=fname, name=f"convergence_{fname}")
    print(f"Đã tạo {len(get_function_names(dim=30))} hình convergence curve (benchmark).")


def _generate_ablation_figure() -> None:
    path = os.path.join("experiments", "results_benchmark", "ablation_summary.csv")
    if not os.path.exists(path):
        print(f"Chưa có {path}, bỏ qua ablation bar chart.")
        return
    df = pd.read_csv(path)
    plot_ablation_bar(df, name="ablation_comparison")
    print("Đã tạo hình ablation_comparison.")


def _generate_fs_figures_if_ready() -> None:
    results_path = os.path.join("experiments", "results_fs", "fs_results.csv")
    if not os.path.exists(results_path):
        print("Chưa có fs_results.csv — bỏ qua hình theo dataset (cần Phase 3 xong).")
        return

    df = pd.read_csv(results_path)
    n_datasets = df["dataset"].nunique()
    n_algorithms = df["algorithm"].nunique()
    if n_datasets < 18 or n_algorithms < 11:
        print(
            f"fs_results.csv mới có {n_datasets}/18 dataset, {n_algorithms}/11 thuật toán "
            "— Phase 3 chưa xong, bỏ qua hình theo dataset (tránh hình dở dang gây hiểu nhầm). "
            "Chạy lại `python -m src.visualization.plots` sau khi Phase 3 hoàn tất."
        )
        return

    for dataset, group in df.groupby("dataset"):
        plot_boxplot_distribution(group, "accuracy", dataset, name=f"boxplot_{dataset}")

    rank_pivot = (
        df.groupby(["dataset", "algorithm"])["accuracy"]
        .mean()
        .unstack("algorithm")
        .rank(axis=1, method="average", ascending=False)
    )
    plot_ranking_heatmap(rank_pivot, name="fs_ranking_heatmap")

    plot_selected_features_bar(df, name="fs_selected_features_ratio")
    print(f"Đã tạo {n_datasets} hình boxplot + 1 heatmap + 1 bar chart (feature selection).")


def main() -> None:
    _generate_benchmark_convergence_figures()
    _generate_ablation_figure()
    _generate_fs_figures_if_ready()


if __name__ == "__main__":
    main()
