import argparse
import os
import json
import numpy as np
from PIL import Image

from eigenx.model_io import load_model
from eigenx.datasets import _to_numpy
from eigenx.align import align_image, default_eye_targets

def parse_args():
    ap = argparse.ArgumentParser("EigenX Batch Reconstruction (First Image of 40 Subjects)")
    ap.add_argument("--data", type=str, required=True, help="ATT-face root directory")
    ap.add_argument("--model", type=str, required=True, help="trained model file (.npz)")
    ap.add_argument("--eye-json", type=str, default=None, help="ATT-eye-location root directory (optional)")
    ap.add_argument("--out", type=str, default="results/face_batch_recon", help="output directory")
    return ap.parse_args()

def _load_and_preprocess(path: str, model: dict, eye_root: str = None) -> np.ndarray:
    """
    加载并预处理图像，逻辑与 myreconstruct.py 保持一致
    """
    img = Image.open(path).convert("L")
    h, w = model["image_shape"]
    
    # 检查模型是否需要眼部对齐
    if model["preprocess"].get("align", "none") == "eye" and eye_root is not None:
        subject = os.path.basename(os.path.dirname(path)) # 例如 s1
        base = os.path.splitext(os.path.basename(path))[0] # 例如 1
        eye_path = os.path.join(eye_root, subject, f"{base}.json")
        
        if os.path.exists(eye_path):
            with open(eye_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            l = tuple(data["centre_of_left_eye"])
            r = tuple(data["centre_of_right_eye"])
            img = align_image(img, l, r, (w, h), default_eye_targets((w, h)))
        else:
            # print(f"Warning: Eye json not found for {path}, resizing directly.")
            img = img.resize((w, h), Image.BILINEAR)
    else:
        img = img.resize((w, h), Image.BILINEAR)
        
    arr = _to_numpy(img, normalize=model["preprocess"].get("normalize", "scale"))
    return arr.reshape(-1)

def convert_to_pil(arr, shape):
    """将向量转换为 0-255 的 PIL 图像"""
    arr = arr.reshape(shape)
    # 简单的 Min-Max 归一化以增强对比度显示
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-5:
        arr = (arr - mn) / (mx - mn) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    args = parse_args()
    
    # 1. 加载模型
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    model = load_model(args.model)
    mu = model["mean"]
    comps = model["components"]
    h, w = model["image_shape"]
    
    # 设定要展示的 PC 数量
    pcs_list = [10, 25, 50, 75]
    
    os.makedirs(args.out, exist_ok=True)
    print(f"Output directory: {args.out}")
    
    # 2. 遍历文件夹寻找 s1...s40 等
    if not os.path.exists(args.data):
        print(f"Error: Data directory {args.data} not found.")
        return

    subjects = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    
    count = 0
    for subject_dir in subjects:
        # 简单过滤，确保是 s 开头的文件夹（如 s1, s2...）
        if not subject_dir.startswith('s'):
             continue

        # 尝试寻找第一张图片 (1.pgm, 1.jpg, 1.png)
        img_path = None
        for ext in [".pgm", ".jpg", ".png", ".bmp"]:
            p = os.path.join(args.data, subject_dir, "1" + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if img_path is None:
            continue
            
        # 3. 预处理
        try:
            x = _load_and_preprocess(img_path, model, args.eye_json)
        except Exception as e:
            print(f"Error processing {subject_dir}: {e}")
            continue

        # 4. 批量重构计算
        outs = []
        for pcs in pcs_list:
            k = min(pcs, comps.shape[1])
            Ck = comps[:, :k]
            # 投影: y = U^T * (x - mean)
            y = Ck.T @ (x - mu)
            # 重构: x_recon = mean + U * y
            xr = mu + Ck @ y
            outs.append(xr)
            
        # 5. 拼接图片
        # 准备图片列表: [原图] + [10 PCs] + [25 PCs] + [50 PCs] + [75 PCs]
        imgs = [convert_to_pil(x, (h, w))] 
        for r in outs:
            imgs.append(convert_to_pil(r, (h, w)))
            
        # 创建长条画布 (宽度 = 单图宽 * 图片数量)
        dst = Image.new("L", (w * len(imgs), h))
        for idx, img in enumerate(imgs):
            dst.paste(img, (idx * w, 0))
            
        # 保存
        base_name = f"{subject_dir}_1_combined.png"
        dst.save(os.path.join(args.out, base_name))
        
        count += 1
        
    print(f"Done! Processed {count} images. Results saved in '{args.out}'.")

if __name__ == "__main__":
    main()