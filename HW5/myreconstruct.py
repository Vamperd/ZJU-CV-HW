import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from eigenx.model_io import load_model
from eigenx.datasets import _to_numpy
from eigenx.align import align_image, default_eye_targets


def parse_args():
    ap = argparse.ArgumentParser("EigenX Face Reconstruction (myreconstruct)")
    ap.add_argument("--input", type=str, required=True, help="input face image file (.pgm/.png/.jpg)")
    ap.add_argument("--model", type=str, required=True, help="trained model file (.npz)")
    ap.add_argument("--eye-json", type=str, default=None, help="ATT-eye-location root directory for alignment")
    ap.add_argument("--pcs", type=int, nargs="*", default=[10, 25, 50, 75], help="PC counts to reconstruct with")
    ap.add_argument("--out", type=str, default="results/faces/recon", help="output directory for reconstructions")
    return ap.parse_args()


def _load_and_preprocess(path: str, model: dict, eye_root: str = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    h, w = model["image_shape"]
    if model["preprocess"].get("align", "none") == "eye" and eye_root is not None:
        subject = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]
        eye_path = os.path.join(eye_root, subject, f"{base}.json")
        if os.path.exists(eye_path):
            import json
            with open(eye_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            l = tuple(data["centre_of_left_eye"])
            r = tuple(data["centre_of_right_eye"])
            img = align_image(img, l, r, (w, h), default_eye_targets((w, h)))
        else:
            img = img.resize((w, h), Image.BILINEAR)
    else:
        img = img.resize((w, h), Image.BILINEAR)
    arr = _to_numpy(img, normalize=model["preprocess"].get("normalize", "scale"))
    return arr.reshape(-1)


def main():
    args = parse_args()
    model = load_model(args.model)
    mu = model["mean"]
    comps = model["components"]
    h, w = model["image_shape"]
    x = _load_and_preprocess(args.input, model, args.eye_json)
    pcs_list = sorted(set([p for p in args.pcs if p >= 1]))
    outs: List[np.ndarray] = []
    for pcs in pcs_list:
        k = min(pcs, comps.shape[1])
        Ck = comps[:, :k]
        y = Ck.T @ (x - mu)
        xr = mu + Ck @ y
        outs.append(xr)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    from eigenx.visualize import save_reconstruction_grid
    save_reconstruction_grid([x], [outs], (h, w), pcs_list, args.out, base)
    print(f"Reconstruction saved under: {args.out}")


if __name__ == "__main__":
    main()

