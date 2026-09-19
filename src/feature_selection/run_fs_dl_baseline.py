from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

from config import KFOLD, KNN_NEIGHBORS, NUM_INDEPENDENT_RUNS, RANDOM_SEED_BASE

PROCESSED_DIR = os.path.join("data", "processed")
FS_MAIN_CSV = os.path.join("experiments", "results_fs", "fs_results.csv")
OUT_DIR = os.path.join("experiments", "results_fs_dl")
OUT_CSV = os.path.join(OUT_DIR, "fs_dl_results.csv")

DEFAULT_DATASETS = ["Zoo", "Sonar", "WDBC", "ColonCancer", "Leukemia"]
EPOCHS = 300
START_TEMP, MIN_TEMP = 10.0, 0.01


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))
    return df.drop(columns=["label"]).to_numpy(dtype=float), df["label"].to_numpy()


def _cv_acc(X: np.ndarray, y: np.ndarray, idx: np.ndarray, seed: int) -> float:
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


class ConcreteSelector(nn.Module):

    def __init__(self, d: int, k: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(k, d))
        nn.init.xavier_normal_(self.logits)
        self.temp = START_TEMP

    def forward(self, x: torch.Tensor, hard: bool) -> torch.Tensor:
        if hard:
            sel = torch.zeros_like(self.logits)
            sel[torch.arange(self.logits.size(0)), self.logits.argmax(1)] = 1.0
        else:
            u = torch.rand_like(self.logits).clamp_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            sel = torch.softmax((self.logits + gumbel) / self.temp, dim=1)
        return x @ sel.t()


class CAE(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.selector = ConcreteSelector(d, k)
        hidden = int(np.clip(k * 2, 32, 256))
        self.decoder = nn.Sequential(
            nn.Linear(k, hidden), nn.LeakyReLU(0.1), nn.Linear(hidden, d)
        )

    def forward(self, x: torch.Tensor, hard: bool) -> torch.Tensor:
        return self.decoder(self.selector(x, hard))


def _cae_select(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X.shape[1]
    k = int(np.clip(k, 1, d - 1))
    Xs = StandardScaler().fit_transform(X)
    xt = torch.tensor(Xs, dtype=torch.float32)
    model = CAE(d, k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    anneal = (MIN_TEMP / START_TEMP) ** (1.0 / max(EPOCHS - 1, 1))
    for ep in range(EPOCHS):
        model.selector.temp = START_TEMP * (anneal ** ep)
        opt.zero_grad()
        recon = model(xt, hard=False)
        loss_fn(recon, xt).backward()
        opt.step()
    with torch.no_grad():
        chosen = torch.unique(model.selector.logits.argmax(1)).cpu().numpy()
    return chosen.astype(int)


def _target_k() -> dict[str, int]:
    df = pd.read_csv(FS_MAIN_CSV)
    rg = df[df.algorithm == "RG-SCSO"]
    col = "n_selected_features" if "n_selected_features" in rg.columns else "n_features"
    return {ds: int(round(g[col].mean())) for ds, g in rg.groupby("dataset")}


def _done_keys() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["dataset"], df["run_id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--runs", type=int, default=NUM_INDEPENDENT_RUNS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        datasets, n_runs = ["Zoo"], 2
    else:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        n_runs = args.runs

    kmap = _target_k()
    os.makedirs(OUT_DIR, exist_ok=True)
    done = _done_keys()
    rows = []
    total = len(datasets) * n_runs
    i = 0
    for ds in datasets:
        X, y = _load(ds)
        k = kmap.get(ds, max(1, X.shape[1] // 2))
        for r in range(n_runs):
            i += 1
            if (ds, r) in done:
                continue
            seed = RANDOM_SEED_BASE + r
            idx = _cae_select(X, k, seed)
            acc = _cv_acc(X, y, idx, seed)
            rows.append({"algorithm": "ConcreteAE", "dataset": ds, "run_id": r,
                         "accuracy": acc, "n_selected_features": int(idx.size),
                         "target_k": int(k)})
            print(f"[{i}/{total}] {ds} run{r}: acc={acc:.4f} "
                  f"nfeat={idx.size} (target k={k})", flush=True)
    if rows:
        new = pd.DataFrame(rows)
        if os.path.exists(OUT_CSV):
            new = pd.concat([pd.read_csv(OUT_CSV), new], ignore_index=True)
        new.to_csv(OUT_CSV, index=False)
    print(f"Xong. Kết quả tại {OUT_CSV}")


if __name__ == "__main__":
    main()
