import os
import csv
import pandas as pd
from tqdm import tqdm
import requests
import subprocess

# 경로 설정
ANNOTATIONS_PATH = "oidv6-train-annotations-bbox.csv"
CLASSES_PATH = "oidv7-class-descriptions-boxable.csv"
IMAGE_ID_LIST_PATH = "stairs_image_ids.txt"
IMAGES_DIR = "images/train"
LABELS_DIR = "labels/train"


# 1. Stairs 클래스 ID 찾기
def get_stairs_label_id():
    with open(CLASSES_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1].lower() == "stairs":
                return row[0]
    raise ValueError("Stairs 클래스 ID를 찾을 수 없습니다.")


# 2. Stairs 객체 이미지 ID 필터링
def filter_stairs_annotations(label_id):
    df = pd.read_csv(ANNOTATIONS_PATH)
    stairs_df = df[df["LabelName"] == label_id]
    return stairs_df


# 3. 이미지 ID 텍스트 파일로 저장
def save_image_id_list(df):
    unique_ids = df["ImageID"].unique()
    with open(IMAGE_ID_LIST_PATH, "w") as f:
        for image_id in unique_ids:
            f.write(f"train/{image_id}\n")
    return unique_ids


def download_downloader_py():
    url = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
    print("🔽 downloader.py 파일 다운로드 중...")
    response = requests.get(url)
    with open("downloader.py", "wb") as f:
        f.write(response.content)


# 4. 이미지 다운로드
def download_images():
    if not os.path.exists("downloader.py"):
        download_downloader_py()
    subprocess.run(
        [
            "python",
            "downloader.py",
            IMAGE_ID_LIST_PATH,
            "--download_folder=" + IMAGES_DIR,
            "--num_processes=5",
        ]
    )


# 5. YOLO 형식 라벨로 변환
def create_yolo_labels(df, image_ids):
    os.makedirs(LABELS_DIR, exist_ok=True)

    grouped = df.groupby("ImageID")
    for image_id, rows in tqdm(grouped, desc="YOLO 라벨 생성"):
        label_path = os.path.join(LABELS_DIR, f"{image_id}.txt")
        with open(label_path, "w") as f:
            for _, row in rows.iterrows():
                # YOLO 포맷 계산: class x_center y_center width height (값은 0~1)
                x_center = (row["XMin"] + row["XMax"]) / 2
                y_center = (row["YMin"] + row["YMax"]) / 2
                width = row["XMax"] - row["XMin"]
                height = row["YMax"] - row["YMin"]
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 메인 실행
def main():
    stairs_id = get_stairs_label_id()
    stairs_df = filter_stairs_annotations(stairs_id)
    image_ids = save_image_id_list(stairs_df)
    download_images()
    create_yolo_labels(stairs_df, image_ids)
    print(f"\n✅ 완료! {len(image_ids)}개의 Stairs 이미지가 준비되었습니다.")


if __name__ == "__main__":
    main()
