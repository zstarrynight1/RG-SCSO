
from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ALPHA = 0.05
TARGET_ALGORITHM = "RG-SCSO"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    total = ranks.sum()
    w_plus = ranks[nonzero > 0].sum()
    w_minus = ranks[nonzero < 0].sum()
    return float((w_plus - w_minus) / total)


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)                          
        adjusted[idx] = min(running_max, 1.0)
    return adjusted

                                                                                 
SOURCES = {
    "benchmark": {
        "results_csv": os.path.join("experiments", "results_benchmark", "benchmark_results.csv"),
        "out_dir": os.path.join("experiments", "results_benchmark"),
        "value_col": "best_fitness",
        "group_col": "function_name",
        "lower_is_better": True,
        "target": "ECL-SCSO",
    },
    "fs": {
        "results_csv": os.path.join("experiments", "results_fs", "fs_results.csv"),
        "out_dir": os.path.join("experiments", "results_fs"),
        "value_col": "accuracy",
        "group_col": "dataset",
        "lower_is_better": False,
        "target": "RG-SCSO",


        "algorithms": ["RG-SCSO", "SCSO", "AOA", "COA", "GWO", "PSO", "RIME"],
    },


    "fs_heldout": {
        "results_csv": os.path.join(
            "experiments", "results_fs_heldout", "fs_heldout_results.csv"
        ),
        "out_dir": os.path.join("experiments", "results_fs_heldout"),
        "value_col": "heldout_accuracy",
        "group_col": "dataset",
        "lower_is_better": False,
        "target": "RG-SCSO",
        "algorithms": ["RG-SCSO", "SCSO", "AOA", "COA", "GWO", "PSO", "RIME"],
    },
}


def paired_wilcoxon_vs_target(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    algorithm_col: str,
    run_col: str,
    target_algorithm: str,
    lower_is_better: bool = True,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    rows = []
    for group_name, group_df in df.groupby(group_col):
        pivot = group_df.pivot(index=run_col, columns=algorithm_col, values=value_col)
        if target_algorithm not in pivot.columns:
            continue
        target_values = pivot[target_algorithm]

        for other_algo in pivot.columns:
            if other_algo == target_algorithm:
                continue
            other_values = pivot[other_algo]
            diff = target_values - other_values

            if np.allclose(diff, 0.0):
                p_value = 1.0
            else:
                try:
                    _, p_value = wilcoxon(target_values, other_values)
                except ValueError:
                                                                               
                    p_value = 1.0

            target_mean, other_mean = target_values.mean(), other_values.mean()
            if lower_is_better:
                target_better = target_mean < other_mean
            else:
                target_better = target_mean > other_mean

            rows.append(
                {
                    "group": group_name,
                    "compared_with": other_algo,
                    "p_value": p_value,
                    f"{target_algorithm}_mean": target_mean,
                    "other_mean": other_mean,
                    "cohens_d": cohens_d(target_values.values, other_values.values),
                    "rank_biserial_r": rank_biserial_r(target_values.values, other_values.values),
                    "target_better": target_better,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result


    result["p_value_holm"] = (
        result.groupby("group")["p_value"].transform(lambda s: holm_correction(s.values))
    )

    def _mark(row: pd.Series) -> str:
        if row["p_value_holm"] < alpha and row["target_better"]:
            return "+"
        if row["p_value_holm"] < alpha and not row["target_better"]:
            return "-"
        return "="

    result["mark"] = result.apply(_mark, axis=1)
    return result.drop(columns=["target_better"])


def friedman_test_and_ranking(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    algorithm_col: str,
    lower_is_better: bool = True,
) -> dict:
    pivot = df.pivot_table(index=group_col, columns=algorithm_col, values=value_col, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")                                              

    samples: Iterable[np.ndarray] = [pivot[col].values for col in pivot.columns]
    statistic, p_value = friedmanchisquare(*samples)

    ranks = pivot.rank(axis=1, method="average", ascending=lower_is_better)
    avg_ranking = ranks.mean(axis=0).sort_values()

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "avg_ranking": avg_ranking,
        "n_groups": pivot.shape[0],
        "n_algorithms": pivot.shape[1],
    }


def run_for_source(source: str) -> None:
    cfg = SOURCES[source]
    if not os.path.exists(cfg["results_csv"]):
        print(f"[{source}] chưa có {cfg['results_csv']}, bỏ qua.")
        return

    df = pd.read_csv(cfg["results_csv"])
    if cfg.get("algorithms"):
        df = df[df["algorithm"].isin(cfg["algorithms"])]
        missing = set(cfg["algorithms"]) - set(df["algorithm"].unique())
        if missing:
            print(f"[{source}] CẢNH BÁO: thiếu thuật toán trong CSV: {sorted(missing)}")
    run_col = "run_id"
    target = cfg.get("target", TARGET_ALGORITHM)

    wilcoxon_df = paired_wilcoxon_vs_target(
        df,
        value_col=cfg["value_col"],
        group_col=cfg["group_col"],
        algorithm_col="algorithm",
        run_col=run_col,
        target_algorithm=target,
        lower_is_better=cfg["lower_is_better"],
    )
    wilcoxon_path = os.path.join(cfg["out_dir"], f"wilcoxon_vs_{target.replace('-', '').lower()}.csv")
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    win = (wilcoxon_df["mark"] == "+").sum()
    lose = (wilcoxon_df["mark"] == "-").sum()
    tie = (wilcoxon_df["mark"] == "=").sum()
    print(
        f"[{source}] Đã ghi {wilcoxon_path} — {target} +{win}/-{lose}/={tie} "
        f"(Holm-corrected, so toàn bộ cặp)."
    )

    friedman = friedman_test_and_ranking(
        df,
        value_col=cfg["value_col"],
        group_col=cfg["group_col"],
        algorithm_col="algorithm",
        lower_is_better=cfg["lower_is_better"],
    )
    ranking_df = friedman["avg_ranking"].rename("avg_rank").reset_index()
    ranking_df.columns = ["algorithm", "avg_rank"]
    ranking_path = os.path.join(cfg["out_dir"], "friedman_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
                                                                         
    pd.DataFrame([{
        "statistic": friedman["statistic"],
        "p_value": friedman["p_value"],
        "n_groups": friedman["n_groups"],
        "n_algorithms": friedman["n_algorithms"],
    }]).to_csv(os.path.join(cfg["out_dir"], "friedman_summary.csv"), index=False)
    print(
        f"[{source}] Friedman chi2={friedman['statistic']:.3f}, p={friedman['p_value']:.3e} "
        f"({friedman['n_groups']} group, {friedman['n_algorithms']} thuật toán). "
        f"Đã ghi {ranking_path}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["benchmark", "fs", "fs_heldout", "all"], default="all"
    )
    args = parser.parse_args()

    sources = list(SOURCES) if args.source == "all" else [args.source]
    for source in sources:
        run_for_source(source)


if __name__ == "__main__":
    main()
