# backend/resize_images.py
from PIL import Image
import os
import re

# 設定圖片資料夾路徑 為此資料夾上一層+ static/images
root_dir = os.path.dirname(os.path.dirname(__file__))
# 再上一層為根目錄
last = os.path.dirname(root_dir)

# root parent 
IMAGE_DIR = os.path.join(last, 'static', 'images')


# IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'static', 'images')

# 設定目標大小
AVATAR_SIZE = (150, 150)
SKILL_SIZE = (40, 40)

def resize_image(image_path, target_size):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")  # 確保所有圖片都有 alpha 通道
            img = img.resize(target_size, Image.ANTIALIAS)
            img.save(image_path)
            print(f"已調整大小: {image_path}")
    except Exception as e:
        print(f"無法調整大小 {image_path}: {e}")

def main():
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(IMAGE_DIR, filename)

            # 判斷圖片類型，擴展正則表達式以支持中文
            if re.match(r'^[\u4e00-\u9fa5\w]+(\.png|\.jpg|\.jpeg)$', filename):
                # Avatar image
                resize_image(file_path, AVATAR_SIZE)
            elif re.match(r'^[\u4e00-\u9fa5\w]+_passive(\.png|\.jpg|\.jpeg)$', filename):
                # Passive skill image
                resize_image(file_path, SKILL_SIZE)
            elif re.match(r'^[\u4e00-\u9fa5\w]+_skill_\d+(\.png|\.jpg|\.jpeg)$', filename):
                # Active skill image
                resize_image(file_path, SKILL_SIZE)
            else:
                print(f"未識別的圖片格式，跳過: {filename}")

if __name__ == "__main__":
    main()
