import os
from typing import Dict, Tuple

import numpy as np


def save_model(
    path: str,
    mu: np.ndarray,
    components_k: np.ndarray,
    singular_values: np.ndarray,
    energy: np.ndarray,
    cum_energy: np.ndarray,
    image_shape: Tuple[int, int],
    preprocess: Dict,
    stats: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=mu,
        components=components_k,
        singular_values=singular_values,
        energy=energy,
        cum_energy=cum_energy,
        image_shape=np.array(image_shape, dtype=np.int32),
        preprocess=np.array([preprocess], dtype=object),
        stats=np.array([stats], dtype=object),
    )


def load_model(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    def _extract_obj(arr):
        try:
            v = arr[0]
            if isinstance(v, np.ndarray):
                return v.item()
            return v
        except Exception:
            return arr.item()
    model = {
        "mean": data["mean"],
        "components": data["components"],
        "singular_values": data["singular_values"],
        "energy": data["energy"],
        "cum_energy": data["cum_energy"],
        "image_shape": tuple(int(x) for x in data["image_shape"].tolist()),
        "preprocess": _extract_obj(data["preprocess"]),
        "stats": _extract_obj(data["stats"]),
    }
    return model
