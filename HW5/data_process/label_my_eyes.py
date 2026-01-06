import os
import json
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# ================= 配置 =================
# 图片所在的文件夹 (处理后的 .pgm 图片)
IMG_DIR = "../myface/output"
# JSON 输出的文件夹
OUT_DIR = "../myface/json_output"
# =======================================

def label_images():
    if not os.path.exists(IMG_DIR):
        print(f"错误: 找不到图片文件夹 {IMG_DIR}")
        return

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # 获取所有图片
    files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.pgm')])
    
    print("="*50)
    print(f"开始标注 {len(files)} 张图片。")
    print("操作说明: ")
    print("1. 图片窗口弹出后。")
    print("2. 鼠标左键点击【画面左侧的眼睛】。")
    print("3. 鼠标左键点击【画面右侧的眼睛】。")
    print("4. 程序会自动保存并切换下一张。")
    print("   (如果点错了，请关闭窗口，程序会重新让您标注这一张)")
    print("="*50)

    # 检查 matplotlib后端，防止不弹窗
    try:
        if matplotlib.get_backend() == 'agg':
            print("警告: 当前 Matplotlib 后端是 'agg' (不显示窗口)。")
            print("尝试切换到 'TkAgg'...")
            matplotlib.use('TkAgg')
    except:
        pass

    for i, filename in enumerate(files):
        img_path = os.path.join(IMG_DIR, filename)
        json_name = f"{os.path.splitext(filename)[0]}.json"
        json_path = os.path.join(OUT_DIR, json_name)

        # 如果已经标注过，跳过
        if os.path.exists(json_path):
            print(f"[{i+1}/{len(files)}] {filename} 已存在，跳过。")
            continue

        img = Image.open(img_path)
        
        while True:
            plt.figure(figsize=(4, 5))
            plt.imshow(img, cmap='gray')
            plt.title(f"{filename}\nClick: Left Eye -> Right Eye")
            plt.axis('off')
            
            print(f"[{i+1}/{len(files)}] 请正在标注 {filename} ...")
            
            # 获取两次点击坐标
            # ginput(n=2) 会等待用户点击 2 次
            # timeout=-1 表示无限等待直到点击
            pts = plt.ginput(n=2, timeout=-1, show_clicks=True)
            plt.close()

            if len(pts) == 2:
                # pts 是列表 [(x1, y1), (x2, y2)]
                eye_left = [pts[0][0], pts[0][1]]  # 画面左边的眼睛
                eye_right = [pts[1][0], pts[1][1]] # 画面右边的眼睛

                # 按照 AT&T 数据集格式构建字典
                # 注意：数据集中的 "left" 通常指 "Subject's Left" (被拍摄者的左眼，即画面右侧)
                # 但这里我们为了对齐算法通用，通常保持 坐标x小的是 left_eye, x大的是 right_eye 即可
                # 或者严格遵循数据集惯例：
                # centre_of_left_eye = 画面中的左侧点 (x坐标较小)
                # centre_of_right_eye = 画面中的右侧点 (x坐标较大)
                
                # 简单的排序确保 x 小的赋给 left
                if eye_left[0] > eye_right[0]:
                    eye_left, eye_right = eye_right, eye_left

                data = {
                    "centre_of_left_eye": eye_left,
                    "centre_of_right_eye": eye_right
                }

                # 保存 JSON
                with open(json_path, 'w') as f:
                    json.dump(data, f)
                print(f"   已保存: {json_name}")
                break
            else:
                print("   点击未完成，请重新标注该图片。")

    print("\n所有图片标注完成！")

if __name__ == "__main__":
    label_images()