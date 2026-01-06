import argparse
import os
import random
import numpy as np
from PIL import Image

from eigenx.model_io import load_model
from eigenx.datasets import _to_numpy

def parse_args():
    ap = argparse.ArgumentParser("EigenX MNIST Random Batch Reconstruction")
    # 注意：这里直接指向存放图片的文件夹，例如 MNIST/digit_7
    ap.add_argument("--data", type=str, required=True, help="Directory containing digit images (e.g. MNIST/digit_7)")
    ap.add_argument("--model", type=str, required=True, help="trained model file (.npz)")
    ap.add_argument("--count", type=int, default=40, help="Number of random images to process")
    ap.add_argument("--out", type=str, default="results/mnist_batch_recon", help="output directory")
    return ap.parse_args()

def convert_to_pil(arr, shape):
    """将向量转换为 0-255 的 PIL 图像"""
    arr = arr.reshape(shape)
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
    h, w = model["image_shape"] # 应该是 28, 28
    
    # 设定要展示的 PC 数量 (MNIST 比较简单，我们可以调整一下梯度，或者保持不变)
    pcs_list = [10, 25, 50, 75]
    
    os.makedirs(args.out, exist_ok=True)
    
    # 2. 获取所有图片并随机采样
    if not os.path.exists(args.data):
        print(f"Error: Data directory {args.data} not found.")
        return

    all_files = [f for f in os.listdir(args.data) if f.lower().endswith(('.png', '.jpg', '.pgm'))]
    total_files = len(all_files)
    
    if total_files == 0:
        print("No images found in data directory.")
        return
        
    # 随机选取 40 张 (如果不足 40 张则全选)
    sample_count = min(args.count, total_files)
    selected_files = random.sample(all_files, sample_count)
    
    print(f"Found {total_files} images. Randomly selected {sample_count} for reconstruction.")

    # 3. 遍历处理
    for idx, filename in enumerate(selected_files):
        img_path = os.path.join(args.data, filename)
        
        # 加载并预处理
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize((w, h), Image.BILINEAR) # 确保尺寸匹配模型
            x = _to_numpy(img, normalize=model["preprocess"].get("normalize", "scale"))
            x = x.reshape(-1)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # 批量重构计算
        outs = []
        for pcs in pcs_list:
            k = min(pcs, comps.shape[1])
            Ck = comps[:, :k]
            # 投影: y = U^T * (x - mean)
            y = Ck.T @ (x - mu)
            # 重构: x_recon = mean + U * y
            xr = mu + Ck @ y
            outs.append(xr)
            
        # 4. 拼接图片
        # 准备图片列表: [原图] + [10 PCs] + [25 PCs] + [50 PCs] + [75 PCs]
        imgs = [convert_to_pil(x, (h, w))] 
        for r in outs:
            imgs.append(convert_to_pil(r, (h, w)))
            
        # 创建长条画布
        dst = Image.new("L", (w * len(imgs), h))
        for i, sub_img in enumerate(imgs):
            dst.paste(sub_img, (i * w, 0))
            
        # 保存 (文件名包含序号，方便查看)
        base_name = f"rand_{idx+1:02d}_{filename}"
        dst.save(os.path.join(args.out, base_name))
        
    print(f"Done! Results saved in '{args.out}'.")

if __name__ == "__main__":
    main()