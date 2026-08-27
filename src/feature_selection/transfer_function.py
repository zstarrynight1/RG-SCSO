"""Sigmoid transfer function — chuyển solution liên tục (output của optimizer,
mỗi chiều đại diện 1 feature) thành binary feature mask.

    S(x) = 1 / (1 + exp(-x))                                   (sigmoid)
    binary_feature = 1 if random() < S(x) else 0                (Eq. PROJECT_SPEC.md mục 5.2)

THIẾT KẾ 2 CHẾ ĐỘ binarize (đều dùng cùng S(x), khác cách quyết định 0/1):
    - `binarize_stochastic`: đúng công thức gốc mục 5.2 (random() < S(x)) —
      dùng TRONG QUÁ TRÌNH tìm kiếm (mỗi lần gọi obj_func), tạo thêm nhiễu
      ngẫu nhiên có lợi cho exploration, giống BPSO gốc (Kennedy & Eberhart).
    - `binarize_threshold`: so sánh S(x) với DIM_BINARY_THRESHOLD (config.py,
      mặc định 0.5) — dùng để chốt feature subset CUỐI CÙNG sau khi tối ưu
      xong (cần kết quả ổn định, reproducible cho bảng số liệu, không phụ
      thuộc lần random() cuối cùng).
    Cả 2 cách đều xuất hiện trong PROJECT_SPEC.md (mục 5.2 cho công thức
    stochastic, mục 2 cho `DIM_BINARY_THRESHOLD`) — tách rõ vai trò để dùng
    được cả 2 mà không mâu thuẫn nhau.
"""

from __future__ import annotations

import numpy as np

from config import DIM_BINARY_THRESHOLD


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binarize_stochastic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = sigmoid(x)
    return (rng.random(x.shape) < s).astype(int)


def binarize_threshold(x: np.ndarray, threshold: float = DIM_BINARY_THRESHOLD) -> np.ndarray:
    s = sigmoid(x)
    return (s >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Họ transfer function cho RG-SCSO (binary-native feature selection)
#
# Chẩn đoán Phase 3: các cải tiến continuous của ECL-SCSO bị "lượng tử hóa mất"
# qua `sigmoid + ngưỡng 0.5` (S-shaped ép mọi chiều về 0/1 theo GIÁ TRỊ tuyệt
# đối của vị trí, không quan tâm trạng thái bit hiện tại). Họ V-shaped
# (Mirjalili & Lewis, "S-shaped versus V-shaped transfer functions for binary
# PSO", Swarm and Evolutionary Computation, 2013) đo XÁC SUẤT LẬT bit tương đối
# so với bit hiện tại — bảo toàn được động lực tìm kiếm trong không gian nhị
# phân, nền tảng cho cơ chế Relevance-Modulated Sensitivity (C1) của RG-SCSO.
# ---------------------------------------------------------------------------


def s_shaped(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer (chính là sigmoid) — giữ tên riêng cho rõ họ hàm."""
    return sigmoid(x)


def v_shaped(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function V(x) = |tanh(x)| (Mirjalili & Lewis, 2013).

    Trả về xác suất LẬT bit (flip) trong [0, 1): |x| lớn -> khả năng đổi trạng
    thái cao (exploration); x≈0 -> gần như giữ nguyên bit (exploitation).

    Args:
        x: Vị trí liên tục của optimizer (np.ndarray, shape bất kỳ).

    Returns:
        np.ndarray cùng shape, mỗi phần tử ∈ [0, 1).
    """
    return np.abs(np.tanh(x))


def binarize_relevance(
    x: np.ndarray,
    prev_bit: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
    gamma: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Relevance-Modulated Sensitivity binarization (C1 của RG-SCSO).

    Diễn giải lại "sensitivity range" của SCSO thành độ nhạy LẬT-BIT theo TỪNG
    feature: xác suất lật cơ sở lấy từ V-shaped của vị trí, rồi điều biến theo
    trường độ-liên-quan ρ. Feature liên quan (ρ cao) -> khó bị bỏ / dễ được
    thêm; feature nhiễu (ρ thấp) -> ngược lại.

    Công thức cho feature j (xem PROJECT_SPEC / plan RG-SCSO):
        p_flip = V(x_j)                                  # xác suất lật cơ sở
        b*_j   = 1 nếu ρ_j > τ else 0                    # bit "được ưu tiên"
        s_j    = |ρ_j − τ| / max(τ, 1−τ)  ∈ [0, 1]        # cường độ ưu tiên
        nếu lật (1 − prev_bit) HƯỚNG VỀ b*_j -> p_flip *= (1 + γ·s_j)
        ngược lại                             -> p_flip *= (1 − γ·s_j)
        lật nếu random() < clip(p_flip, 0, 1)

    γ = 0 làm hàm suy biến về V-shaped thuần (không dùng relevance) — dùng cho
    ablation "NoRMS" để chứng minh C1 load-bearing.

    Tổng quát hóa ngưỡng τ (threshold sensitivity, Q1 review): `strength` chia
    cho max(τ, 1−τ) — KHÔNG chỉ đổi 0.5→τ trong công thức 2·|ρ−0.5| gốc — vì
    khoảng cách xa nhất có thể từ τ tới 2 đầu mút [0,1] là max(τ, 1−τ), không
    còn là 0.5 khi τ≠0.5; chia cho đúng khoảng cách này giữ s_j∈[0,1] với MỌI
    τ. Tại τ=0.5 công thức quy giản CHÍNH XÁC về bản gốc (max(0.5,0.5)=0.5 nên
    |ρ−0.5|/0.5 = 2·|ρ−0.5|) — tái tạo đúng 100% mọi kết quả đã có, không phải
    một xấp xỉ.

    Args:
        x: Vị trí liên tục sau bước di chuyển SCSO, shape (n_features,).
        prev_bit: Binary mask hiện tại của cá thể, shape (n_features,), giá trị 0/1.
        rho: Trường độ-liên-quan ρ ∈ [0, 1], shape (n_features,).
        rng: numpy Generator (reproducible, seed từ optimizer).
        gamma: Cường độ điều biến relevance (0 = tắt).
        threshold: τ — ngưỡng phân tách bit "được ưu tiên" (0.5 = mặc định gốc,
            xem sweep τ∈{0.4,0.5,0.6} trong threshold-sensitivity study).

    Returns:
        Binary mask mới, np.ndarray shape (n_features,), giá trị 0/1 (int).
    """
    p_flip = v_shaped(x)
    preferred = (rho > threshold).astype(int)
    norm = max(threshold, 1.0 - threshold)
    strength = np.abs(rho - threshold) / norm
    flip_bit = 1 - prev_bit
    toward_preferred = flip_bit == preferred
    factor = np.where(toward_preferred, 1.0 + gamma * strength, 1.0 - gamma * strength)
    p_flip = np.clip(p_flip * factor, 0.0, 1.0)
    do_flip = rng.random(x.shape) < p_flip
    return np.where(do_flip, flip_bit, prev_bit).astype(int)
