# CV HW5: Eigenfaces (PCA) Implementation

本项目实现了基于主成分分析（PCA）的特征脸（Eigenfaces）算法，用于人脸重建与识别。项目支持 AT&T 人脸数据集、MNIST 手写数字数据集，以及用户自定义人脸数据的处理、训练与可视化重建。此外，还包含一个简单的 Web 前端演示应用。

---

## 📂 项目文件结构

```text
HW5/
├── apps/ (或 HTML/)       # Web 应用前端与后端
│   ├── app.py             # Flask 后端服务
│   └── index.html         # 前端演示页面
├── data/                  # 数据集根目录
│   ├── ATT-face/          # AT&T 人脸图片 (s1-s40, s41)
│   ├── ATT-eye-location/  # 眼部坐标 JSON 文件
│   └── MNIST/             # MNIST 手写数字数据集
├── data_output/           # 批量处理与输出脚本
│   ├── reconstruct_batch.py       # 批量重构人脸脚本
│   └── reconstruct_mnist_batch.py # 批量重构 MNIST 脚本
├── data_process/          # 数据预处理工具
│   ├── prepare_mnist.py      # 下载/提取 MNIST 数据
│   ├── preprocess_my_face.py # 自定义照片预处理 (缩放/补黑边)
│   └── label_my_eyes.py      # 自定义照片眼部标注工具
├── eigenx/                # 核心算法库 (PCA, IO, Align)
├── models/                # 训练好的模型文件 (.npz)
├── myface/                # 自定义人脸的原始素材与临时输出
├── results/               # 运行结果输出目录
├── mytrain.py             # 主训练脚本
├── myreconstruct.py       # 单张图片重构测试脚本
└── requirements.txt       # (可选) 依赖库列表
```


## 🚀 快速开始
### 0. 环境依赖
确保安装了以下 Python 库：

### 1. AT&T 人脸模型训练与重构
此步骤将训练标准人脸模型，并生成所有人的重构对比图。
Step 1: 训练模型 读取 ATT-face，启用眼部对齐（Align），保留 95% 的总能量。

Step 2: 批量验证 (生成 s1-s40 的重构对比图) 遍历所有受试者，对每个人的代表性图片（默认为第1张）进行 10/25/50/75 PCs 的重构对比。

结果查看：进入 face_batch_recon 查看生成的拼接对比图。

### 2. MNIST 手写数字实验
此步骤针对单一数字（如 "7"）进行 PCA 分析。
Step 1: 数据准备 如果 MNIST 为空，运行此脚本下载并提取数字 "7" 的图片：

Step 2: 训练模型 注意：MNIST 图片尺寸为 28x28，必须显式指定 --resize 参数，且不需要眼部对齐。

Step 3: 随机批量重构 从数据集中随机抽取 40 张数字 "7" 进行重构测试。

### 3. 添加自定义人脸 (s41)
将你自己的照片加入数据集，训练包含你特征的模型。
准备照片：用手机拍摄约 10 张大头照（建议背景纯净），放入 raw 文件夹。
自动预处理：运行脚本将图片缩放并填充至 92x112（不会变形）：
标注眼睛：运行交互式脚本，在弹出的窗口中依次点击左眼和右眼：
合并数据：
将 output 中的所有 .pgm 图片移动到 s41。
将 json_output 中的所有 .json 文件移动到 s41。
重新训练：再次运行 "AT&T 人脸模型" 的训练与重构命令即可（脚本会自动识别新加入的 s41）。
### 4. Web 在线演示
启动一个简单的网页，拖拽上传图片查看实时重构效果。
配置后端： 打开 app.py，确认 MODEL_PATH 指向你训练好的模型（例如 ../models/face_model.npz）。

启动服务：

访问前端： 在浏览器打开 http://127.0.0.1:5000 (如果未自动跳转，直接打开 index.html 即可)。

## ⚙️ 核心脚本参数说明
mytrain.py (训练)
- data: [必须] 数据集根目录。
- model: [必须] 输出模型文件路径 (.npz)。
- p: [必须] 保留主成分的能量百分比 (0-100)，推荐 95。
- resize: [可选] 强制缩放输入图像大小 (AT&T 默认不填，MNIST 需填 28 28)。
- align: [可选] 标志位，是否启用眼部对齐 (旋转+缩放)。
- eye-json: [可选] 眼部坐标数据目录 (配合 --align 使用)。

reconstruct_batch.py (批量重构)
- data: [必须] 原始图片目录。
- model: [必须] 已训练的模型路径。
- out: [可选] 结果图片输出目录。