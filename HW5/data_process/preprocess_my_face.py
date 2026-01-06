import os
from PIL import Image, ImageOps

def process_my_faces(source_dir, target_dir, target_size=(92, 112)):
    """
    处理人脸图片：
    为了防止人脸被裁切掉，采用 '缩放 + 填充黑边' (Pad) 的方式，
    而不是 '居中裁剪' (Crop)。这样能保留完整的原图视野。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")

    files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
    
    if not files:
        print(f"No images found in {source_dir}!")
        return

    print(f"Found {len(files)} images. Re-processing with Padding...")

    target_w, target_h = target_size

    for i, filename in enumerate(files):
        img_path = os.path.join(source_dir, filename)
        try:
            with Image.open(img_path) as img:
                # 1. 转为灰度
                img = img.convert('L')
                
                # 2. 保持比例缩放到能完全放入 target_size 中
                # Image.ratio 算法:
                # 如果图片是 1000x2000 (高瘦), 目标 92x112
                # 它会缩放到 56x112 (高度填满，宽度不足)
                img.thumbnail((target_w, target_h), Image.LANCZOS)
                
                # 3. 创建纯黑背景
                new_img = Image.new("L", (target_w, target_h), 0)
                
                # 4. 将缩放后的图粘贴到中间
                paste_x = (target_w - img.width) // 2
                paste_y = (target_h - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # 5. 保存
                save_name = f"{i+1}.pgm" 
                save_path = os.path.join(target_dir, save_name)
                new_img.save(save_path)
                print(f"Saved: {save_path}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Done! Images have been padded and resized.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "myface", "raw")
    output_path = os.path.join(base_dir, "myface", "output")
    
    if not os.path.exists(input_path):
        print(f"Error: Raw directory not found: {input_path}")
    else:
        process_my_faces(input_path, output_path)