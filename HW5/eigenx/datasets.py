import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

from .align import align_image, default_eye_targets


def iter_face_files(root_dir: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for subject in sorted(os.listdir(root_dir)):
        s_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(s_dir):
            continue
        for name in sorted(os.listdir(s_dir)):
            if name.lower().endswith((".pgm", ".png", ".jpg", ".jpeg")):
                items.append((subject, name, os.path.join(s_dir, name)))
    return items


def _load_eye_json(eye_root: Optional[str], subject: str, img_name: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if eye_root is None:
        return None
    base = os.path.splitext(img_name)[0]
    eye_path = os.path.join(eye_root, subject, f"{base}.json")
    if not os.path.exists(eye_path):
        return None
    with open(eye_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    l = data.get("centre_of_left_eye", None)
    r = data.get("centre_of_right_eye", None)
    if l is None or r is None:
        return None
    return (float(l[0]), float(l[1])), (float(r[0]), float(r[1]))


def _to_numpy(img: Image.Image, normalize: str) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if normalize == "scale":
        arr = arr / 255.0
    elif normalize == "none":
        pass
    else:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr


def load_faces(
    img_root: str,
    eye_root: Optional[str],
    resize: Tuple[int, int] = (112, 92),
    use_eye_align: bool = True,
    normalize: str = "scale",
) -> Tuple[np.ndarray, Tuple[int, int], List[Dict]]:
    items = iter_face_files(img_root)
    w, h = resize
    X_list: List[np.ndarray] = []
    meta_list: List[Dict] = []
    for subject, name, path in items:
        try:
            img = Image.open(path).convert("L")
        except Exception:
            continue
        if use_eye_align and eye_root is not None:
            eyes = _load_eye_json(eye_root, subject, name)
            if eyes is not None:
                img = align_image(img, eyes[0], eyes[1], (w, h), default_eye_targets((w, h)))
            else:
                img = img.resize((w, h), Image.BILINEAR)
        else:
            img = img.resize((w, h), Image.BILINEAR)
        arr = _to_numpy(img, normalize)
        X_list.append(arr.reshape(-1))
        meta_list.append({"subject": subject, "name": name, "path": path})
    if not X_list:
        raise RuntimeError("No valid face images loaded")
    X = np.stack(X_list, axis=0)
    return X, (h, w), meta_list

