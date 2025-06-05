import os
import cv2
import pandas as pd
from pathlib import Path

# === 路径配置 ===
data_dir = Path('data')
samm_dir = data_dir / 'SAMM'
output_dir = data_dir / 'processed_SAMM'
cascade_path = data_dir / 'haarcascade_frontalface_default.xml'
excel_path = data_dir / 'SAMM_Micro_FACS_Codes_v2.xlsx'

# === 加载人脸检测器 ===
face_cascade = cv2.CascadeClassifier(str(cascade_path))

# === 加载标注文件 ===
df = pd.read_excel(excel_path, header=13)

# === 处理每一行 ===
for idx, row in df.iterrows():
    subject = str(row['Subject']).zfill(3)  # 补零如 '006'
    folder_name = row['Filename']           # 如 '006_1_2'
    input_folder = samm_dir / subject / folder_name

    if not input_folder.exists():
        print(f"[警告] 输入文件夹不存在: {input_folder}")
        continue

    # 输出路径
    output_folder = output_dir / subject / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # 处理该文件夹下所有图片
    for img_file in input_folder.glob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        # 读取图像
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[跳过] 无法读取图像: {img_file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"[未检测到人脸] {img_file}")
            continue

        # 取第一张检测到的人脸
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        # 调整尺寸为128x128
        face_resized = cv2.resize(face, (128, 128))

        # 保存（保持原始文件名）
        output_path = output_folder / img_file.name
        cv2.imwrite(str(output_path), face_resized)
        print(f"[已保存] {output_path}")
