"""Baseline filter/embedded kinh điển (Q1 review Priority 1) — kiểm định câu hỏi
cốt lõi mà tiêu đề "parsimonious feature selection" phải trả lời: một phương
pháp rẻ hơn RẤT NHIỀU so với swarm wrapper 15.000 evaluations có đạt độ nhỏ
gọn/accuracy tương đương hay không?

NĂM phương pháp, tất cả đánh giá bằng CÙNG protocol với bài chính (KNN k=5,
5-fold CV, StandardScaler fit riêng mỗi fold):

    MI-threshold : ρ_j = MI(X_j;y)/H(y) > 0.5 — đúng ngưỡng trung tính RG-SCSO
                   dùng, KHÔNG search, chi phí gần như 0.
    mRMR         : greedy Max-Relevance Min-Redundancy (relevance = MI, độ dư
                   thừa = |Pearson correlation| giữa các feature đã chọn),
                   dừng bằng CV-forward-stepwise (patience trên validation
                   accuracy) — KHÔNG cố định K trước.
    ReliefF      : dùng lại _relieff_prior() (đã có trong relevance.py cho thí
                   nghiệm robustness), chọn tăng dần theo điểm ReliefF, dừng
                   bằng CÙNG cơ chế CV-forward-stepwise như mRMR.
    LASSO        : LogisticRegressionCV(penalty="l1"), feature có hệ số khác 0
                   (tối đa trị tuyệt đối qua các lớp với multiclass).
    SFS          : sklearn SequentialFeatureSelector (forward), giới hạn
                   n_features_to_select để khả thi trên tập gene-expression
                   nhiều chiều (xem MAX_SFS_FEATURES).

30 run độc lập (seed = BASE + run_id) để có mean±std như bài chính — dù các
phương pháp này không stochastic search, CV-fold splitting + MI estimator vẫn
có phương sai đáng báo cáo. Quy mô PILOT trên 5 dataset đại diện (khớp
ablation/robustness) để khả thi về compute; mở rộng ra 18 dataset là bước
tiếp theo nếu tín hiệu sơ bộ ủng hộ đưa vào bài chính thức.

Output: experiments/results_fs_classic/fs_classic_results.csv
Chạy:   .venv/bin/python -m src.feature_selection.run_fs_classic_baselines
        [--smoke] [--datasets ...] [--runs N] [--methods ...]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import KFOLD, KNN_NEIGHBORS, NUM_INDEPENDENT_RUNS, RANDOM_SEED_BASE
from src.feature_selection.fitness import evaluate_binary_mask
from src.feature_selection.relevance import _relieff_prior

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("experiments", "results_fs_classic")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "fs_classic_results.csv")

DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
METHODS = ("MI-threshold", "mRMR", "ReliefF-baseline", "LASSO", "SFS")

# Trần feature cho SFS (greedy O(d) fit MỖI bước -> O(d x K) fit tổng cộng);
# không giới hạn thì ColonCancer/Leukemia (2000-3571 feature) bất khả thi
# trong ngân sách 1 lượt chạy. Trần chọn xấp xỉ mức RG-SCSO thực tế chọn.
MAX_SFS_FEATURES = 25
# Pool ứng viên cho SFS trên dataset nhiều chiều (xem docstring _select_sfs).
SFS_CANDIDATE_POOL = 80
# Trần cho vòng lặp mRMR/ReliefF-greedy (patience-based nhưng vẫn cần chặn
# trên để không chạy quá độ dài trên dataset siêu nhiều chiều).
MAX_GREEDY_FEATURES = 150
PATIENCE = 8
TOL = 1e-3


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _cv_acc(X: np.ndarray, y: np.ndarray, idx: np.ndarray, seed: int) -> float:
    """5-fold CV KNN accuracy trên tập con feature `idx`, scaler fit riêng mỗi
    fold — khớp fitness.py::_cv_accuracy của bài chính (không leakage)."""
    if idx.size == 0:
        return 0.0
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()),
                          ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))])
        pipe.fit(X[tr][:, idx], y[tr])
        accs.append(pipe.score(X[te][:, idx], y[te]))
    return float(np.mean(accs))


def _greedy_forward(scores_order: np.ndarray, X: np.ndarray, y: np.ndarray, seed: int,
                     max_features: int = MAX_GREEDY_FEATURES) -> np.ndarray:
    """Thêm feature lần lượt theo `scores_order` (đã sắp giảm dần độ ưu tiên),
    dừng khi CV accuracy không cải thiện > TOL trong PATIENCE bước liên tiếp
    (forward-stepwise CV, không cố định K trước — khớp tinh thần mRMR/ReliefF
    gốc: chọn TỚI KHI hết lợi ích biên, không phải một K tùy ý)."""
    cap = min(max_features, len(scores_order))
    selected: list[int] = []
    best_acc = 0.0
    best_selected: list[int] = []
    no_improve = 0
    for j in scores_order[:cap]:
        selected.append(int(j))
        acc = _cv_acc(X, y, np.array(selected), seed)
        if acc > best_acc + TOL:
            best_acc = acc
            best_selected = list(selected)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    return np.array(best_selected if best_selected else selected[:1])


def _select_mi_threshold(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_y = float(-np.sum(p * np.log(p + 1e-12)))
    mi = mutual_info_classif(X, y, random_state=seed)
    rho = np.clip(mi / h_y, 0.0, 1.0) if h_y > 0 else np.full(X.shape[1], 0.5)
    idx = np.flatnonzero(rho > 0.5)
    return idx if idx.size else np.array([int(np.argmax(rho))])


def _select_mrmr(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Greedy mRMR-FCQ: relevance = MI(X_j;y); độ dư thừa = |Pearson corr| với
    feature ĐÃ chọn (proxy chuẩn cho continuous feature, O(d^2) chấp nhận
    được qua ma trận, tránh O(d^2) lời gọi MI riêng lẻ bất khả thi trên
    ColonCancer/Leukemia)."""
    d = X.shape[1]
    relevance = mutual_info_classif(X, y, random_state=seed)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = set(range(d)) - set(selected)
    best_acc = _cv_acc(X, y, np.array(selected), seed)
    best_selected = list(selected)
    no_improve = 0
    cap = min(MAX_GREEDY_FEATURES, d)
    while remaining and len(selected) < cap:
        red = np.array([np.mean(np.abs(corr[j, selected])) for j in remaining])
        rel = np.array([relevance[j] for j in remaining])
        mrmr_score = rel - red
        pick = list(remaining)[int(np.argmax(mrmr_score))]
        selected.append(pick)
        remaining.discard(pick)
        acc = _cv_acc(X, y, np.array(selected), seed)
        if acc > best_acc + TOL:
            best_acc, best_selected, no_improve = acc, list(selected), 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    return np.array(best_selected)


def _select_relieff(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    w = _relieff_prior(X, y, seed)  # điểm 0.5-centered, cao = liên quan hơn
    order = np.argsort(-w)
    return _greedy_forward(order, X, y, seed)


def _select_lasso(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    n_classes = len(np.unique(y))
    solver = "liblinear" if n_classes == 2 else "saga"
    clf = LogisticRegressionCV(
        penalty="l1", solver=solver, cv=min(KFOLD, 3), max_iter=2000,
        random_state=seed, n_jobs=1,
    )
    clf.fit(Xs, y)
    coef = np.abs(clf.coef_)
    weight = coef.max(axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
    idx = np.flatnonzero(weight > 1e-8)
    return idx if idx.size else np.array([int(np.argmax(weight))])


def _select_sfs(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Forward SFS. sklearn's SequentialFeatureSelector re-scores EVERY
    remaining candidate at each step (O(k x d x cv) fits) — infeasible as-is
    on gene-expression data (d up to 3571): 40 steps x ~3500 candidates x
    3-fold would be ~400k KNN fits per run. We pre-filter the candidate POOL
    to the top-`SFS_CANDIDATE_POOL` features by MI (a standard, disclosed
    practical trick for making greedy wrapper selection tractable on high-d
    data) and run genuine forward SFS only within that pool; this is a
    protocol choice reported in the paper, not a silent shortcut."""
    n_target = max(1, min(MAX_SFS_FEATURES, X.shape[1] - 1))
    pool_size = min(SFS_CANDIDATE_POOL, X.shape[1])
    if X.shape[1] > pool_size:
        mi = mutual_info_classif(X, y, random_state=seed)
        pool = np.argsort(-mi)[:pool_size]
    else:
        pool = np.arange(X.shape[1])
    n_target = min(n_target, pool.size - 1) if pool.size > 1 else 1
    sfs = SequentialFeatureSelector(
        KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS),
        n_features_to_select=n_target, direction="forward",
        cv=min(KFOLD, 3), n_jobs=1,
    )
    sfs.fit(StandardScaler().fit_transform(X[:, pool]), y)
    return pool[np.flatnonzero(sfs.get_support())]


_SELECTORS = {
    "MI-threshold": _select_mi_threshold,
    "mRMR": _select_mrmr,
    "ReliefF-baseline": _select_relieff,
    "LASSO": _select_lasso,
    "SFS": _select_sfs,
}


def _existing_keys() -> set[tuple[str, str, int]]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["algorithm"], df["dataset"], df["run_id"]))


def _run_single(task: dict) -> dict:
    method, ds, run_id = task["algorithm"], task["dataset"], task["run_id"]
    X, y = _load(ds)
    seed = RANDOM_SEED_BASE + run_id
    t0 = time.time()
    idx = _SELECTORS[method](X, y, seed)
    mask = np.zeros(X.shape[1], dtype=int)
    mask[idx] = 1
    final = evaluate_binary_mask(mask, X, y, seed=seed)
    return {
        "algorithm": method, "dataset": ds, "run_id": run_id,
        "fitness": final["fitness"], "accuracy": final["accuracy"],
        "n_selected_features": final["n_selected_features"],
        "n_total_features": final["n_total_features"],
        "runtime_seconds": time.time() - t0,
    }


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="1 run, Zoo, tất cả method — kiểm tra wiring.")
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--methods", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.smoke:
        datasets, methods, n_runs, workers = ["Zoo"], list(METHODS), 1, 1
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        methods = args.methods.split(",") if args.methods else list(METHODS)
        n_runs, workers = args.runs, os.cpu_count()

    done = set() if args.smoke else _existing_keys()
    tasks = [
        {"algorithm": m, "dataset": ds, "run_id": r}
        for m in methods for ds in datasets for r in range(n_runs)
        if (m, ds, r) not in done
    ]
    total = len(methods) * len(datasets) * n_runs
    print(f"Classic baselines: {len(methods)} method x {len(datasets)} dataset x "
          f"{n_runs} run = {total} task ({len(done)} đã xong, chạy {len(tasks)}).")

    if args.smoke:
        rows = [_run_single(t) for t in tasks]
        print(pd.DataFrame(rows).to_string())
        print("\nSMOKE OK — wiring chạy, mask hợp lệ. KHÔNG dùng làm kết quả.")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classic"):
            try:
                _append_rows([fut.result()])
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                print(f"[LỖI] {t['algorithm']} x {t['dataset']} x run{t['run_id']}: {exc}")

    print(f"Xong. Kết quả tại {RESULTS_CSV}")


if __name__ == "__main__":
    main()
