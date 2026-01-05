import os
import gzip
import numpy as np
from PIL import Image
import struct

def read_idx_images(path):
    """读取 MNIST 图像文件 (支持 .gz 或 解压后的二进制)"""
    open_func = gzip.open if path.endswith('.gz') else open
    print(f"Reading images from {path}...")
    with open_func(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return data

def read_idx_labels(path):
    """读取 MNIST 标签文件 (支持 .gz 或 解压后的二进制)"""
    open_func = gzip.open if path.endswith('.gz') else open
    print(f"Reading labels from {path}...")
    with open_func(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        return data

def main():
    # 1. 设置路径
    # 假设脚本在 HW5/ 目录下，数据在 HW5/MNIST/raw
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "MNIST", "raw")
    
    # 定义可能的文件名 (优先找解压后的，找不到找 .gz)
    possible_img_names = ["train-images.idx3-ubyte", "train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"]
    possible_lbl_names = ["train-labels.idx1-ubyte", "train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"]
    
    img_path = None
    lbl_path = None

    # 查找图像文件
    for name in possible_img_names:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            img_path = p
            break
            
    # 查找标签文件
    for name in possible_lbl_names:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            lbl_path = p
            break
            
    if not img_path or not lbl_path:
        print(f"Error: Could not find MNIST files in {raw_dir}")
        print(f"Please ensure files are named like: {possible_img_names[0]} or {possible_img_names[-1]}")
        return

    # 2. 读取数据
    try:
        images = read_idx_images(img_path)
        labels = read_idx_labels(lbl_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    
    # 3. 筛选数字 7 并保存
    digit_target = 7
    # 输出目录: MNIST/digit_7
    output_dir = os.path.join(base_dir, "MNIST", f"digit_{digit_target}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting digit {digit_target} to {output_dir}...")
    
    count = 0
    for i in range(len(labels)):
        if labels[i] == digit_target:
            # 保存为 PNG
            img = Image.fromarray(images[i])
            # 文件名: 00001.png
            save_path = os.path.join(output_dir, f"{count+1:05d}.png")
            img.save(save_path)
            count += 1
            
    print(f"Done! Saved {count} images of digit {digit_target}.")

if __name__ == "__main__":
    main()