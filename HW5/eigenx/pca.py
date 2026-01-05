from typing import Tuple, Dict

import numpy as np


def pca_train(X: np.ndarray) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt.T
    variances = (S ** 2)
    total_var = variances.sum()
    energy = variances / (total_var + 1e-12)
    cum_energy = np.cumsum(energy)
    return {
        "mean": mu.astype(np.float32),
        "components": components.astype(np.float32),
        "singular_values": S.astype(np.float32),
        "energy": energy.astype(np.float32),
        "cum_energy": cum_energy.astype(np.float32),
    }


def select_k_by_energy(cum_energy: np.ndarray, percent: float, max_k: int = None) -> int:
    target = max(0.0, min(100.0, percent)) / 100.0
    idx = int(np.searchsorted(cum_energy, target) + 1)
    if max_k is not None:
        idx = min(idx, max_k)
    return max(1, idx)


def project(x: np.ndarray, mu: np.ndarray, components_k: np.ndarray) -> np.ndarray:
    return components_k.T @ (x - mu)


def reconstruct(y: np.ndarray, mu: np.ndarray, components_k: np.ndarray) -> np.ndarray:
    return mu + components_k @ y

