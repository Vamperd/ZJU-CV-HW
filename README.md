# 浙江大学计算机视觉课程作业 (ZJU Computer Vision Homework)

本仓库用于存放浙江大学计算机视觉课程的作业代码、实验报告及相关资源。包含从基础图像处理到深度学习应用的 5 次作业。

## 📂 作业列表与内容

### [HW1] 图像基础处理 (Image Basics)
*   **语言**: C++
*   **主要文件**: `main.cpp`, `CMakeLists.txt`
*   **内容**: 实现了基础的图像读取与处理操作。
*   **资源**: `resources/TRY.png`

### [HW2] 边缘与形状检测 (Edge & Shape Detection)
*   **语言**: C++
*   **主要文件**: `main.cpp`
*   **内容**: 
    *   实现了边缘检测 (Edge Detection)。
    *   实现了霍夫变换 (Hough Transform) 检测直线和圆。
    *   应用场景包括硬币检测 (`coin.png`)、高速公路车道线检测 (`highway.png`) 等。
*   **结果展示**: 见 `HW2/result` 文件夹，包含检测到的圆、边缘及最终叠加结果。

### [HW3] 图像拼接与全景图 (Image Stitching & Panorama)
*   **语言**: C++
*   **主要文件**: `main.cpp`
*   **内容**: 
    *   实现了基于特征点的图像配准与拼接。
    *   处理了多张图片的融合，生成全景图像。
    *   测试数据包括 Yosemite 风景图及其他测试图片。
*   **结果展示**: `HW3/result/result.jpg`

### [HW4] 卷积神经网络分类 (CNN Classification)
*   **语言**: Python (PyTorch/TensorFlow)
*   **主要文件**: 
    *   `lenet5.py`, `MNIST_train.py`: 使用 LeNet-5 在 MNIST 数据集上进行手写数字识别。
    *   `resnet.py`, `CIFAR_train.py`: 使用 ResNet 在 CIFAR-10 数据集上进行图像分类。
*   **实验分析**: 
    *   对比了不同优化策略（如 Cutout, Data Enhancement）对训练曲线的影响。
    *   训练曲线图见 `HW4/result` 文件夹。

### [HW5] 特征脸与图像重建 (Eigenfaces & Reconstruction)
*   **语言**: Python
*   **核心算法**: 主成分分析 (PCA)
*   **主要文件**: 
    *   `mytrain.py`: 训练 PCA 模型，提取特征向量 (Eigenvectors)。
    *   `myreconstruct.py`: 使用特征向量重建人脸图像。
    *   `eigenx/`: 包含 PCA 核心实现、数据对齐与可视化工具。
*   **内容**: 
    *   **人脸重建**: 使用 PCA 对人脸数据集进行降维与重建 (`results/faces`).
    *   **MNIST 重建**: 对 MNIST 数字 '7' 进行 PCA 分析与重建 (`results/mnist_7`).
    *   **Web 展示**: 包含一个简单的 HTML 前端 (`HTML/index.html`) 用于展示结果。

---

## 🛠️ 环境依赖与编译
本项目包含 C++ 和 Python 两种实现。
鉴于仓库大小与历年作业调整，本仓库不存储所有数据集，请依照需求自行download