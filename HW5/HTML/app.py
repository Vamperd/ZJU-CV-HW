import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

# 引入您现有的逻辑
from eigenx.model_io import load_model
from eigenx.datasets import _to_numpy
from eigenx.align import align_image, default_eye_targets

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# --- 全局加载模型 (避免每次请求都加载，提高速度) ---
# 请根据实际情况修改模型路径，这里默认加载人脸模型
MODEL_PATH = "../models/face_model_aligned.npz" 
# 如果想支持 MNIST，可以加载两个模型，或者通过前端参数选择
model_data = None

def init_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model_data = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")

# --- 核心处理逻辑 (复用 reconstruct_batch.py 的逻辑) ---
def process_image(pil_img, pcs_list=[10, 25, 50, 75]):
    if model_data is None:
        raise ValueError("Model not loaded")

    mu = model_data["mean"]
    comps = model_data["components"]
    h, w = model_data["image_shape"]

    # 1. 预处理 (这里简化处理，暂不支持眼部对齐，直接缩放)
    img = pil_img.convert("L").resize((w, h), Image.BILINEAR)
    x = _to_numpy(img, normalize=model_data["preprocess"].get("normalize", "scale"))
    x = x.reshape(-1)

    # 2. 批量重构
    outs = []
    for pcs in pcs_list:
        k = min(pcs, comps.shape[1])
        Ck = comps[:, :k]
        y = Ck.T @ (x - mu)
        xr = mu + Ck @ y
        outs.append(xr)

    # 3. 拼接图片
    def convert_to_pil(arr):
        arr = arr.reshape((h, w))
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-5:
            arr = (arr - mn) / (mx - mn) * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    imgs = [convert_to_pil(x)] + [convert_to_pil(r) for r in outs]
    
    # 创建长条图
    dst = Image.new("L", (w * len(imgs), h))
    for idx, sub_img in enumerate(imgs):
        dst.paste(sub_img, (idx * w, 0))
    
    return dst

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        img = Image.open(file.stream)
        result_img = process_image(img)
        
        # 将结果图片转为字节流返回
        img_io = io.BytesIO()
        result_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)