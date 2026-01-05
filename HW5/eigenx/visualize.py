import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_image(arr: np.ndarray, shape: Tuple[int, int]) -> Image.Image:
    h, w = shape
    a = arr.reshape(h, w)
    a_min = a.min()
    a_max = a.max()
    if a_max - a_min < 1e-8:
        a_norm = np.zeros_like(a)
    else:
        a_norm = (a - a_min) / (a_max - a_min)
    a_img = (a_norm * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a_img, mode="L")


def save_eigenfaces(components: np.ndarray, shape: Tuple[int, int], out_dir: str, top_k: int = 10) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    k = min(top_k, components.shape[1])
    for i in range(k):
        comp = components[:, i]
        img = _to_image(comp, shape)
        p = os.path.join(out_dir, f"eigenface_{i+1:02d}.png")
        img.save(p)
        paths.append(p)
    return paths


def save_reconstruction_grid(
    originals: List[np.ndarray],
    reconstructions: List[List[np.ndarray]],
    shape: Tuple[int, int],
    pcs_list: List[int],
    out_dir: str,
    base_name: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    for idx, orig in enumerate(originals):
        orig_img = _to_image(orig, shape)
        for j, pcs in enumerate(pcs_list):
            rec_img = _to_image(reconstructions[idx][j], shape)
            p = os.path.join(out_dir, f"{base_name}_{idx:02d}_pcs{pcs}.png")
            rec_img.save(p)
            saved.append(p)
        p0 = os.path.join(out_dir, f"{base_name}_{idx:02d}_orig.png")
        orig_img.save(p0)
        saved.append(p0)
    return saved


def save_energy_curve(cum_energy: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save as simple CSV to avoid plotting dependencies
    data = np.stack([np.arange(1, len(cum_energy) + 1), cum_energy], axis=1)
    np.savetxt(out_path, data, fmt="%.6f", delimiter=",", header="k,cum_energy", comments="")


def save_energy_curve_png(cum_energy: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w, h = 800, 600
    margin = 80
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x0, y0 = margin, h - margin
    x1, y1 = w - margin, margin
    draw.line([(x0, y0), (x1, y0)], fill=(0, 0, 0), width=2)
    draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)
    n = len(cum_energy)
    if n <= 1:
        img.save(out_path)
        return
    xs = np.linspace(x0, x1, n)
    ys = y0 - (y0 - y1) * np.clip(cum_energy, 0.0, 1.0)
    pts = list(zip(xs.tolist(), ys.tolist()))
    draw.line(pts, fill=(34, 139, 230), width=3)
    for i in range(0, n, max(1, n // 10)):
        draw.ellipse([pts[i][0] - 3, pts[i][1] - 3, pts[i][0] + 3, pts[i][1] + 3], fill=(34, 139, 230))
    for t in [0.0, 0.5, 1.0]:
        y = y0 - (y0 - y1) * t
        draw.line([(x0 - 5, y), (x0 + 5, y)], fill=(0, 0, 0), width=1)
        draw.text((x0 - 50, y - 8), f"{t:.1f}", fill=(0, 0, 0))
    draw.text((x1 - 20, y0 + 10), "k", fill=(0, 0, 0))
    draw.text((x0 - 20, y1 - 30), "energy", fill=(0, 0, 0))
    img.save(out_path)
