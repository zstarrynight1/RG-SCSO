"""Tải và chuẩn hóa 18 dataset UCI/Kaggle/GitHub thành format chuẩn cho Phase 3:
cột feature đứng trước, cột cuối cùng tên `label` (số nguyên 0..K-1).

NGUỒN DỮ LIỆU (đã xác minh thủ công, KHÔNG tự ý thay dataset khác khi gặp
nguồn không rõ ràng — đã hỏi và xác nhận với người dùng cho 2 trường hợp lệch
so với mô tả trong PROJECT_SPEC.md mục 5.1):

    13 dataset lấy trực tiếp từ UCI ML Repository chính thức qua package
    `ucimlrepo`: BreastEW, WDBC, SpectEW, HeartDisease, Parkinsons,
    Lymphography, IonosphereEW, Sonar, Vote, Zoo, TicTacToe, KrVsKpEW,
    GermanCredit.

    Diabetes (Pima Indians): KHÔNG còn trên UCI chính thức (đã bị gỡ).
    Dùng mirror GitHub `jbrownlee/Datasets` (768x8, khớp đúng dataset gốc),
    spec mục 5.1 đã cho phép nguồn "UCI/Kaggle".

    M-of-n, WaveformEW (bản 40-feature — bản UCI API chỉ có bản 21-feature):
    lấy từ `github.com/thieu1995/MHA-FS/data` (repo đã archive, tác giả của
    `mealpy`/`mafese`).

    ColonCancer, Leukemia: PLOS ONE supplementary data, DOI
    10.1371/journal.pone.0246039 (qua figshare API). Leukemia ở nguồn này
    chỉ có 3571 gene (không phải ~7129 như mô tả gốc trong spec) — ĐÃ HỎI
    và được xác nhận dùng bản 3571 gene này.

QUY TẮC XỬ LÝ CHUNG (áp dụng đồng nhất cho mọi dataset, để dễ giải thích
trong phần Methodology):
    1. Cột nào toàn bộ giá trị thiếu (bug parsing của 1 vài nguồn, ví dụ cột
       "no. of nodes in" của Lymphography qua ucimlrepo) bị loại bỏ TRƯỚC.
    2. Sau đó, loại bỏ các DÒNG còn giá trị thiếu ('?', NaN).
    3. Cột categorical (object dtype) -> label-encode thành số nguyên, GIỮ
       NGUYÊN số lượng feature gốc (không one-hot, để khớp số chiều đã công
       bố trong literature, ví dụ Vote=16, Zoo=16, KrVsKpEW=35).
    4. Cột nhãn -> label-encode thành số nguyên 0..K-1.
    5. KHÔNG scale numeric feature ở bước này — việc chuẩn hóa (StandardScaler)
       được thực hiện BÊN TRONG mỗi fold của k-fold CV ở `fitness.py`, để
       tránh data leakage giữa train/test (lý do quan trọng cần ghi rõ khi
       viết Methodology, vì nhiều bài báo cùng dạng mắc lỗi leakage này).
    6. Heart Disease: nhãn gốc "num" có 5 mức (0-4, mức độ bệnh) -> binarize
       thành 0 (không bệnh) / 1 (có bệnh, num>=1) theo đúng convention phổ
       biến trong literature feature-selection nhị phân.

Output: 1 file CSV mỗi dataset trong `data/processed/<TenDataset>.csv`.

Chạy: python -m src.feature_selection.prepare_datasets
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

PROCESSED_DIR = os.path.join("data", "processed")
RAW_DIR = os.path.join("data", "raw")

PLOS_FIGSHARE_URLS = {
    "ColonCancer": "https://ndownloader.figshare.com/files/26216253",
    "Leukemia": "https://ndownloader.figshare.com/files/26216250",
}
GITHUB_RAW_URLS = {
    "Diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
    "M-of-n": "https://raw.githubusercontent.com/thieu1995/MHA-FS/main/data/M-of-n.csv",
    "WaveformEW": "https://raw.githubusercontent.com/thieu1995/MHA-FS/main/data/WaveformEW.csv",
}


def _dedupe_columns(columns: pd.Index) -> list[str]:
    """1 vài dataset UCI (Parkinsons, Lymphography) có tên cột feature trùng
    nhau (ví dụ 'MDVP:Jitter' xuất hiện 2 lần) — khiến `X[col]` trả về
    DataFrame thay vì Series. Đổi tên trùng thành 'col', 'col.1', 'col.2'...
    giống cách pandas tự xử lý khi đọc CSV có header trùng."""
    counts: dict[str, int] = {}
    result = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
            result.append(col)
        else:
            counts[col] += 1
            result.append(f"{col}.{counts[col]}")
    return result


def _clean_and_encode(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Áp dụng quy tắc xử lý chung (mục docstring trên), trả về DataFrame
    cuối cùng (feature... + label), sẵn sàng ghi ra CSV."""
    X = X.copy()
    X.columns = _dedupe_columns(X.columns)

    # 1. Loại cột toàn NaN (bug parsing 1 vài nguồn)
    all_na_cols = X.columns[X.isna().all()]
    X = X.drop(columns=all_na_cols)

    # 2. Loại dòng còn giá trị thiếu
    mask_valid = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X.loc[mask_valid].reset_index(drop=True), y.loc[mask_valid].reset_index(drop=True)

    # 3. Label-encode cột categorical (giữ nguyên số lượng feature)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 4. Label-encode nhãn
    label = LabelEncoder().fit_transform(y.astype(str))

    out = X.astype(float)
    out["label"] = label
    return out


def _load_ucirepo_dataset(uid: int, binarize_threshold: int | None = None) -> pd.DataFrame:
    ds = fetch_ucirepo(id=uid)
    X = ds.data.features
    y = ds.data.targets[ds.data.targets.columns[0]]
    if binarize_threshold is not None:
        y = (pd.to_numeric(y, errors="coerce") >= binarize_threshold).astype(int)
    return _clean_and_encode(X, y)


def _load_csv_url(url: str, header: int | None) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), header=header)


def _load_diabetes() -> pd.DataFrame:
    df = _load_csv_url(GITHUB_RAW_URLS["Diabetes"], header=None)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return _clean_and_encode(X, y)


def _load_m_of_n() -> pd.DataFrame:
    df = _load_csv_url(GITHUB_RAW_URLS["M-of-n"], header=None)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return _clean_and_encode(X, y)


def _load_waveform_ew() -> pd.DataFrame:
    df = _load_csv_url(GITHUB_RAW_URLS["WaveformEW"], header=None)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return _clean_and_encode(X, y)


def _load_plos_microarray(name: str) -> pd.DataFrame:
    resp = requests.get(PLOS_FIGSHARE_URLS[name], timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.drop(columns=["samples"])  # cột ID mẫu, không phải feature
    label_col = "response" if "response" in df.columns else "Response"
    X, y = df.drop(columns=[label_col]), df[label_col]
    return _clean_and_encode(X, y)


# (tên dataset chuẩn hóa) -> hàm loader
DATASET_LOADERS = {
    "BreastEW": lambda: _load_ucirepo_dataset(15),
    "WDBC": lambda: _load_ucirepo_dataset(17),
    "SpectEW": lambda: _load_ucirepo_dataset(95),
    "HeartDisease": lambda: _load_ucirepo_dataset(45, binarize_threshold=1),
    "Parkinsons": lambda: _load_ucirepo_dataset(174),
    "Diabetes": _load_diabetes,
    "Lymphography": lambda: _load_ucirepo_dataset(63),
    "ColonCancer": lambda: _load_plos_microarray("ColonCancer"),
    "Leukemia": lambda: _load_plos_microarray("Leukemia"),
    "IonosphereEW": lambda: _load_ucirepo_dataset(52),
    "Sonar": lambda: _load_ucirepo_dataset(151),
    "Vote": lambda: _load_ucirepo_dataset(105),
    "Zoo": lambda: _load_ucirepo_dataset(111),
    "M-of-n": _load_m_of_n,
    "TicTacToe": lambda: _load_ucirepo_dataset(101),
    "KrVsKpEW": lambda: _load_ucirepo_dataset(22),
    "WaveformEW": _load_waveform_ew,
    "GermanCredit": lambda: _load_ucirepo_dataset(144),
}


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    rows_summary = []
    for name, loader in DATASET_LOADERS.items():
        print(f"Đang xử lý {name} ...", end=" ", flush=True)
        try:
            df = loader()
        except Exception as exc:  # noqa: BLE001 — muốn thấy rõ dataset nào lỗi
            print(f"LỖI: {exc}")
            continue
        out_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)
        n_features = df.shape[1] - 1
        n_classes = df["label"].nunique()
        rows_summary.append((name, df.shape[0], n_features, n_classes))
        print(f"OK — {df.shape[0]} mẫu, {n_features} feature, {n_classes} lớp")

    print("\n=== TỔNG KẾT ===")
    print(f"{'Dataset':<15}{'#mẫu':>8}{'#feature':>10}{'#lớp':>8}")
    for name, n, f, c in rows_summary:
        print(f"{name:<15}{n:>8}{f:>10}{c:>8}")


if __name__ == "__main__":
    main()
