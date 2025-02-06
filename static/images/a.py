import os
from PIL import Image

def main():
    # 取得當前目錄
    current_dir = os.getcwd()
    
    # 遍歷目錄中的所有檔案
    for filename in os.listdir(current_dir):
        # 判斷檔案是否以 .png 結尾（忽略大小寫）
        if filename.lower().endswith('.png'):
            try:
                # 開啟圖片檔案
                with Image.open(filename) as img:
                    # 將圖片尺寸調整為 150x150 像素
                    resized_img = img.resize((150, 150), Image.ANTIALIAS)
                    # 儲存調整後的圖片，覆蓋原檔案
                    resized_img.save(filename)
                    print(f"已調整圖片尺寸：{filename}")
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤：{e}")

if __name__ == '__main__':
    main()