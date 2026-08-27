"""Kiểm định thống kê cho Phase 4 (PROJECT_SPEC.md mục 6.1):

    - Wilcoxon signed-rank test: so ECL-SCSO với từng thuật toán còn lại,
      TRÊN TỪNG dataset/hàm benchmark riêng. Output bảng p-value + dấu
      "+/-/=" (ECL-SCSO thắng/thua/ngang theo ý nghĩa thống kê alpha=0.05).
    - Friedman test + average ranking: xếp hạng tất cả thuật toán qua toàn
      bộ dataset/hàm benchmark.

LƯU Ý PHƯƠNG PHÁP (quan trọng, cần ghi vào Methodology của bài báo):
    Spec gốc ghi "Wilcoxon rank-sum test", nhưng thiết kế thí nghiệm của ta
    PAIRED theo run_id (mọi thuật toán ở cùng run_id dùng CÙNG
    RANDOM_SEED_BASE + run_id) — đây là paired design, nên kiểm định đúng về
    mặt thống kê là Wilcoxon SIGNED-RANK test (`scipy.stats.wilcoxon`), KHÔNG
    phải rank-sum/Mann-Whitney (vốn dành cho 2 mẫu độc lập không ghép cặp).
    Đây cũng là lựa chọn phổ biến trong hầu hết các bài báo metaheuristic
    feature-selection cùng dạng (dù nhiều bài gọi tên không chính xác là
    "rank-sum test" trong văn bản nhưng code thực tế dùng `scipy.stats.wilcoxon`
    paired). Khi viết Methodology, nên trích dẫn rõ là Wilcoxon signed-rank
    test (Wilcoxon, 1945), paired theo seed.

    Friedman test dùng `scipy.stats.friedmanchisquare` (đã kiểm định tốt,
    thay cho việc tự code lại — spec cho phép "scipy.stats hoặc tự code").
"""

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
    """Cohen's d cho thiết kế PAIRED (a, b khớp cặp theo run/seed).

    d = mean(a − b) / std(a − b). Diễn giải độ lớn hiệu ứng (Cohen, 1988):
    |d|<0.2 negligible, 0.2 small, 0.5 medium, 0.8 large. Bắt buộc theo
    Q1_BLUEPRINT (effect size không được thiếu bên cạnh p-value).

    Args:
        a: Mẫu của thuật toán target, shape (n_runs,).
        b: Mẫu của thuật toán so sánh, cùng thứ tự cặp với a.

    Returns:
        Cohen's d (float). 0.0 nếu độ lệch chuẩn của hiệu = 0.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r — effect size đi kèm Wilcoxon
    signed-rank (Kerby, 2014).

    r = (W+ − W−) / sum(ranks) ∈ [−1, 1]. Dương = a có xu hướng lớn hơn b.
    Bỏ qua các cặp có hiệu = 0 (đúng quy ước Wilcoxon signed-rank).

    Args:
        a: Mẫu target, shape (n_runs,).
        b: Mẫu so sánh, cùng thứ tự cặp.

    Returns:
        Rank-biserial r (float). 0.0 nếu mọi cặp có hiệu = 0.
    """
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
    """Hiệu chỉnh đa so sánh Holm–Bonferroni (Holm, 1979) — step-down, kiểm soát
    family-wise error rate, mạnh hơn Bonferroni thuần.

    Sắp p tăng dần; p_(k) điều chỉnh = (m − k) · p_(k); ép đơn điệu không giảm
    và cắt trần tại 1.0. Trả về theo ĐÚNG thứ tự đầu vào.

    Args:
        p_values: Mảng p-value thô, shape (m,).

    Returns:
        np.ndarray p-value đã hiệu chỉnh, shape (m,), cùng thứ tự đầu vào.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)  # ép đơn điệu không giảm
        adjusted[idx] = min(running_max, 1.0)
    return adjusted

# (đường dẫn kết quả thô, value_col fitness/accuracy, group_col, lower_is_better)
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
        # Bộ so sánh KHÓA ở EXPERIMENT_PROTOCOL.md Mục 2 (7 thuật toán). CSV còn
        # lẫn kết quả cũ Phase 3 (ECL-SCSO, GA, WOA, HHO, SSA, OOA) tái dùng chung
        # compute — lọc về đúng bộ pre-registered, KHÔNG phải cherry-pick.
        "algorithms": ["RG-SCSO", "SCSO", "AOA", "COA", "GWO", "PSO", "RIME"],
    },
    # R3b — fold-honest held-out generalization (EXPERIMENT_PROTOCOL.md change log
    # 2026-07-08). Metric of record = accuracy trên held-out-20 (MI prior + search
    # + CV-fitness CHỈ trên train-80). CÙNG bộ 7 thuật toán khóa như "fs"; filter
    # isin() tự bỏ algo chưa chạy (staged: RG-SCSO/SCSO/AOA trước), Friedman +
    # Wilcoxon chạy trên đúng những algo đang có trong CSV.
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
    """So `target_algorithm` với từng thuật toán khác, RIÊNG cho mỗi group
    (function_name/dataset). Trả về DataFrame cột:
        group, compared_with, p_value, target_mean, other_mean, mark
    `mark`: "+" (target thắng có ý nghĩa thống kê), "-" (thua), "=" (không
    khác biệt có ý nghĩa thống kê, hoặc 2 mẫu giống hệt nhau).
    """
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
                    # toàn bộ cặp có hiệu = 0 (scipy raise lỗi thay vì trả p=1)
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

    # Holm hiệu chỉnh đa so sánh: family = các cặp trong CÙNG 1 group (mỗi group
    # so target với (n_algorithms − 1) thuật toán khác).
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
    """Friedman test (omnibus, H0: mọi thuật toán có performance như nhau qua
    các group) + average ranking. Mỗi group (function/dataset) là 1 block,
    giá trị mỗi block = mean của `value_col` qua các run (đã tổng hợp sẵn ở
    summary_stats.csv/fs_summary.csv).

    Trả về dict: {statistic, p_value, avg_ranking (Series, thấp=tốt nếu
    lower_is_better), n_groups, n_algorithms}.
    """
    pivot = df.pivot_table(index=group_col, columns=algorithm_col, values=value_col, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")  # Friedman cần đầy đủ thuật toán ở mọi block

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
    # Lưu chi2/p để bản thảo đọc trực tiếp (single source, không gõ tay).
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
