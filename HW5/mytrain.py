import argparse
import os
from typing import Tuple

import numpy as np

from eigenx.datasets import load_faces
from eigenx.pca import pca_train, select_k_by_energy
from eigenx.model_io import save_model
from eigenx.visualize import save_eigenfaces, save_energy_curve, save_energy_curve_png


def parse_args():
    ap = argparse.ArgumentParser("EigenX Face Training (mytrain)")
    ap.add_argument("--data", type=str, required=True, help="ATT-face root directory")
    ap.add_argument("--eye-json", type=str, default=None, help="ATT-eye-location root directory")
    ap.add_argument("--p", type=float, required=True, help="energy percentage (0-100)")
    ap.add_argument("--model", type=str, required=True, help="output model file path (.npz)")
    ap.add_argument("--resize", type=int, nargs=2, default=[112, 92], help="target size W H")
    ap.add_argument("--align", action="store_true", help="use eye alignment")
    ap.add_argument("--max-k", type=int, default=None, help="max principal components cap")
    ap.add_argument("--out", type=str, default="results/faces", help="output directory for visuals")
    return ap.parse_args()


def main():
    args = parse_args()
    w, h = args.resize
    X, shape, metas = load_faces(args.data, args.eye_json, (w, h), use_eye_align=args.align, normalize="scale")
    res = pca_train(X)
    cum_energy = res["cum_energy"]
    k = select_k_by_energy(cum_energy, args.p, args.max_k)
    components_k = res["components"][:, :k]
    os.makedirs(args.out, exist_ok=True)
    feats_dir = os.path.join(args.out, "features")
    energy_path_csv = os.path.join(args.out, "energy_curve.csv")
    energy_path_png = os.path.join(args.out, "energy_curve.png")
    save_eigenfaces(components_k, shape, feats_dir, top_k=10)
    save_energy_curve(cum_energy, energy_path_csv)
    save_energy_curve_png(cum_energy, energy_path_png)
    preprocess = {
        "gray": True,
        "resize": (h, w),
        "normalize": "scale",
        "align": "eye" if args.align and args.eye_json is not None else "none",
    }
    stats = {"n_samples": int(X.shape[0]), "k_selected": int(k), "p_target": float(args.p)}
    save_model(
        args.model,
        res["mean"],
        components_k,
        res["singular_values"],
        res["energy"],
        cum_energy,
        shape,
        preprocess,
        stats,
    )
    print(f"Training done. Selected k={k} PCs for target energy {args.p}%.")
    print(f"Model saved to: {args.model}")
    print(f"Eigenfaces saved to: {feats_dir}")
    print(f"Energy curve saved to: {energy_path_png}")


if __name__ == "__main__":
    main()
