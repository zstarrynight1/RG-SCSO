
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
    X = X.copy()
    X.columns = _dedupe_columns(X.columns)

                                                    
    all_na_cols = X.columns[X.isna().all()]
    X = X.drop(columns=all_na_cols)

                                    
    mask_valid = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X.loc[mask_valid].reset_index(drop=True), y.loc[mask_valid].reset_index(drop=True)

                                                                   
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                          
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
    df = df.drop(columns=["samples"])                                  
    label_col = "response" if "response" in df.columns else "Response"
    X, y = df.drop(columns=[label_col]), df[label_col]
    return _clean_and_encode(X, y)


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
