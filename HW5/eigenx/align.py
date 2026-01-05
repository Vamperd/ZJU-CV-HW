import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image


def default_eye_targets(size: Tuple[int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    w, h = size
    cy = 0.38 * h
    dx = 0.25 * w
    left = (w / 2.0 - dx, cy)
    right = (w / 2.0 + dx, cy)
    return left, right


def _similarity_transform_from_2points(
    l_src: Tuple[float, float],
    r_src: Tuple[float, float],
    l_dst: Tuple[float, float],
    r_dst: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    v_src = np.array([r_src[0] - l_src[0], r_src[1] - l_src[1]], dtype=np.float64)
    v_dst = np.array([r_dst[0] - l_dst[0], r_dst[1] - l_dst[1]], dtype=np.float64)

    norm_src = np.linalg.norm(v_src)
    norm_dst = np.linalg.norm(v_dst)
    if norm_src < 1e-6 or norm_dst < 1e-6:
        A = np.eye(2, dtype=np.float64)
        t = np.zeros(2, dtype=np.float64)
        return A, t

    s = norm_dst / norm_src
    ang_src = math.atan2(v_src[1], v_src[0])
    ang_dst = math.atan2(v_dst[1], v_dst[0])
    theta = ang_dst - ang_src
    c = math.cos(theta)
    s_th = math.sin(theta)
    R = np.array([[c, -s_th], [s_th, c]], dtype=np.float64)
    A = s * R
    t = np.array([l_dst[0], l_dst[1]], dtype=np.float64) - A @ np.array([l_src[0], l_src[1]], dtype=np.float64)
    return A, t


def _pil_affine_from_At(A: np.ndarray, t: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    return (
        float(A_inv[0, 0]),
        float(A_inv[0, 1]),
        float(t_inv[0]),
        float(A_inv[1, 0]),
        float(A_inv[1, 1]),
        float(t_inv[1]),
    )


def align_image(
    img: Image.Image,
    left_eye: Tuple[float, float],
    right_eye: Tuple[float, float],
    size: Tuple[int, int],
    eye_targets: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Image.Image:
    img = img.convert("L")
    w, h = size
    if eye_targets is None:
        eye_targets = default_eye_targets((w, h))
    l_t, r_t = eye_targets
    A, t = _similarity_transform_from_2points(left_eye, right_eye, l_t, r_t)
    params = _pil_affine_from_At(A, t)
    aligned = img.transform((w, h), Image.AFFINE, params, resample=Image.BILINEAR)
    return aligned

