"""Trường độ-liên-quan (relevance field) ρ cho RG-SCSO — Online Relevance
Learning (C2).

ρ_j ∈ [0, 1] là "mức độ liên quan" của feature j với nhãn, hợp nhất từ 2 nguồn:

    1. Prior TĨNH (filter): information gain ratio dựa trên mutual information
       I(feature_j; y) chuẩn hóa theo entropy nhãn H(y). Tính MỘT lần từ (X, y),
       rẻ, không phụ thuộc quá trình tìm kiếm — bơm tri thức bài toán ngay từ đầu.

    2. Điểm học ONLINE (credit assignment): EMA của "công trạng" mỗi feature suy
       ra từ độ thay đổi fitness của các bước di chuyển được chấp nhận. Feature
       xuất hiện trong mask giúp fitness TỐT lên -> điểm dương; trong mask làm
       fitness XẤU đi -> điểm âm. Bầy đàn tự học feature nào quan trọng cho CHÍNH
       dataset đang giải (tương tác feature mà prior tĩnh không thấy được).

WHY tách static/online: khởi đầu online = 0 nên ρ ≈ prior (an toàn, có định
hướng); càng về sau online càng điều chỉnh ρ theo bằng chứng thực nghiệm thu
được trong lúc tối ưu. Gỡ online (dùng prior tĩnh thuần) là ablation "NoORL" để
chứng minh C2 load-bearing.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Các phương pháp prior hỗ trợ (dùng cho thí nghiệm robustness rev #6: kiểm tra
# lợi ích RG-SCSO không phải artifact riêng của MI + KNN "hợp rơ").
# "shuffled_mi"/"inverted_mi" (Q1 review Priority 3): can thiệp nhân quả lên
# CHÍNH prior MI thật — permute (xáo trộn gán ρ giữa các feature, giữ nguyên
# phân phối biên) hoặc đảo (1-ρ, đổi feature "liên quan" thành "nhiễu" và
# ngược lại) — để kiểm định liệu hiệu năng/độ nhỏ gọn có SỤP ĐỔ khi ánh xạ
# relevance bị phá vỡ hay không (bằng chứng nhân quả thật, không chỉ tương
# quan như size-fair enrichment).
PRIOR_METHODS = ("mi", "relieff", "fisher", "shuffled_mi", "inverted_mi")


def relevance_prior(
    X: np.ndarray, y: np.ndarray, seed: int, method: str = "mi"
) -> np.ndarray:
    """Prior độ-liên-quan tĩnh ρ_static ∈ [0, 1] cho từng feature.

    Ba filter score, đều ánh xạ về thang [0, 1] với 0.5 = trung tính (RMS không
    điều biến), >0.5 = ưu tiên GIỮ, <0.5 = ưu tiên BỎ:

        - "mi"      (mặc định, dùng trong bài chính): information gain ratio
          ρ_j = clip(I(X_j;y)/H(y), 0, 1). Vì I ≤ H(y) nên tỉ số ∈ [0, 1] —
          bounded, diễn giải được, không cần min-max tùy dataset.
        - "relieff" : ReliefF (Kononenko 1994), nhạy tương tác feature; trọng số
          có dấu (âm = gây nhiễu) nên map 0.5 + w/(2·max|w|) để giữ 0 ↦ 0.5.
        - "fisher"  : Fisher score (giữa/trong lớp), ≥ 0 nên min-max về [0, 1].

    Args:
        X: Ma trận feature, shape (n_samples, n_features).
        y: Nhãn, shape (n_samples,).
        seed: Seed cho thành phần ngẫu nhiên (MI estimator; lấy mẫu ReliefF).
        method: Một trong PRIOR_METHODS.

    Returns:
        np.ndarray shape (n_features,), mỗi phần tử ∈ [0, 1].
    """
    if method not in PRIOR_METHODS:
        raise ValueError(f"method phải thuộc {PRIOR_METHODS}, nhận '{method}'.")

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_y = float(-np.sum(p * np.log(p + 1e-12)))  # entropy nhãn (nats)
    if h_y <= 0.0:  # chỉ 1 class -> mọi feature "vô can", trả 0.5 trung tính
        return np.full(X.shape[1], 0.5)

    if method == "mi":
        mi = mutual_info_classif(X, y, random_state=seed)
        return np.clip(mi / h_y, 0.0, 1.0)
    if method == "relieff":
        return _relieff_prior(X, y, seed)
    if method == "fisher":
        return _fisher_prior(X, y)
    if method == "shuffled_mi":
        mi = mutual_info_classif(X, y, random_state=seed)
        rho = np.clip(mi / h_y, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        return rng.permutation(rho)
    # "inverted_mi": đảo cực (1-ρ) — feature liên quan thật thành "nhiễu" theo
    # RMS và ngược lại; nếu cơ chế thật sự nhân quả, đảo cực phải làm subset
    # phình to VÀ/HOẶC accuracy sụp, không chỉ trung tính.
    mi = mutual_info_classif(X, y, random_state=seed)
    rho = np.clip(mi / h_y, 0.0, 1.0)
    return 1.0 - rho


def _minmax01(v: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector về [0, 1]; nếu hằng số trả 0.5 (trung tính)."""
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.full_like(v, 0.5, dtype=float)
    return (v - lo) / (hi - lo)


def _fisher_prior(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher score: (phương sai giữa lớp) / (phương sai trong lớp), theo feature.

    F_j = Σ_c n_c (μ_{c,j} − μ_j)² / Σ_c n_c σ²_{c,j}. Điểm cao = phân tách lớp
    tốt. Fisher ≥ 0 nên min-max về [0, 1] (thang tương đối trong cùng dataset).
    """
    classes = np.unique(y)
    mu = X.mean(axis=0)
    num = np.zeros(X.shape[1], dtype=float)
    den = np.zeros(X.shape[1], dtype=float)
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        num += nc * (Xc.mean(axis=0) - mu) ** 2
        den += nc * Xc.var(axis=0)
    fisher = num / (den + 1e-12)
    return _minmax01(fisher)


def _relieff_prior(
    X: np.ndarray, y: np.ndarray, seed: int, n_neighbors: int = 10, max_samples: int = 300
) -> np.ndarray:
    """ReliefF (Kononenko 1994) → ρ ∈ [0, 1].

    Với m instance lấy mẫu ngẫu nhiên (cap `max_samples` để O(m·n·d) khả thi
    trên dataset nhiều mẫu), cập nhật trọng số mỗi feature bằng hiệu chuẩn hóa
    tới k near-hit (cùng lớp) và k near-miss theo từng lớp khác (có trọng số
    theo prior lớp). Trọng số dương = feature phân biệt lớp; âm = gây nhiễu.
    Map 0.5 + w/(2·max|w|) để 0 ↦ 0.5 (trung tính), giữ đúng ngữ nghĩa RMS.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    span = X.max(axis=0) - X.min(axis=0)
    span[span < 1e-12] = 1.0
    Xn = (X - X.min(axis=0)) / span  # [0,1]-scale để diff công bằng giữa feature

    classes, counts = np.unique(y, return_counts=True)
    prior = {c: cnt / n for c, cnt in zip(classes, counts)}

    m = min(max_samples, n)
    sample_idx = rng.choice(n, size=m, replace=False)
    k = max(1, min(n_neighbors, n - 1))

    w = np.zeros(d, dtype=float)
    for i in sample_idx:
        xi, yi = Xn[i], y[i]
        dist = np.abs(Xn - xi).sum(axis=1)  # Manhattan trên thang [0,1]
        dist[i] = np.inf
        # near-hits (cùng lớp): trừ đi hiệu (feature ổn định trong lớp -> tốt)
        same = np.where(y == yi)[0]
        same = same[same != i]
        if same.size:
            hits = same[np.argsort(dist[same])[:k]]
            w -= np.abs(Xn[hits] - xi).sum(axis=0) / (m * hits.size)
        # near-misses (mỗi lớp khác): cộng hiệu có trọng số prior
        denom = 1.0 - prior[yi]
        for c in classes:
            if c == yi:
                continue
            oth = np.where(y == c)[0]
            if oth.size == 0:
                continue
            miss = oth[np.argsort(dist[oth])[:k]]
            weight_c = (prior[c] / denom) if denom > 1e-12 else 0.0
            w += weight_c * np.abs(Xn[miss] - xi).sum(axis=0) / (m * miss.size)

    scale = float(np.max(np.abs(w)))
    if scale < 1e-12:
        return np.full(d, 0.5)
    return np.clip(0.5 + w / (2.0 * scale), 0.0, 1.0)


class RelevanceField:
    """Giữ ρ_static + ρ_online và cung cấp ρ hợp nhất `combined()`.

    ρ_online là EMA (exponential moving average) của tín hiệu credit-assignment,
    bounded trong [-1, 1] nhờ `tanh` reward -> `combined` luôn ∈ [0, 1].

    Attributes:
        rho_static: Prior tĩnh ∈ [0, 1], shape (n_features,).
        online: Điểm học online (EMA) ∈ [-1, 1], shape (n_features,), khởi tạo 0.
    """

    def __init__(
        self,
        rho_static: np.ndarray,
        ema_lambda: float = 0.9,
        w_online: float = 0.3,
        delta_scale: float = 0.01,
    ) -> None:
        """
        Args:
            rho_static: Prior tĩnh từ `relevance_prior`.
            ema_lambda: Hệ số EMA (giữ 90% lịch sử mỗi cập nhật).
            w_online: Trọng số đóng góp của online vào ρ hợp nhất.
            delta_scale: Thang chuẩn hóa độ thay đổi fitness trước khi qua tanh
                (fitness delta điển hình ~0.01; +0.01 -> reward ≈ tanh(1) ≈ 0.76).
        """
        self.rho_static = np.asarray(rho_static, dtype=float)
        self.online = np.zeros_like(self.rho_static)
        self.ema_lambda = ema_lambda
        self.w_online = w_online
        self.delta_scale = delta_scale

    def update(self, mask: np.ndarray, fitness_delta: float) -> None:
        """Cập nhật ρ_online từ 1 bước di chuyển.

        Args:
            mask: Binary mask của cá thể SAU khi di chuyển, shape (n_features,).
            fitness_delta: fitness_cũ − fitness_mới (>0 nghĩa là cải thiện, vì
                fitness càng nhỏ càng tốt).
        """
        reward = float(np.tanh(fitness_delta / self.delta_scale))
        signal = mask.astype(float) * reward  # feature được chọn nhận reward có dấu
        self.online = self.ema_lambda * self.online + (1.0 - self.ema_lambda) * signal

    def combined(self) -> np.ndarray:
        """ρ hợp nhất ∈ [0, 1]: clip(ρ_static + w_online · ρ_online, 0, 1).

        Khởi đầu (online = 0) -> combined = ρ_static (prior thuần); càng về sau
        càng dịch theo bằng chứng online.

        Returns:
            np.ndarray shape (n_features,), mỗi phần tử ∈ [0, 1].
        """
        return np.clip(self.rho_static + self.w_online * self.online, 0.0, 1.0)
