import os
import json
import cv2
import numpy as np
from glob import glob

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# 設定你的資料夾路徑
DATA_DIR = "/home/yuyun/Desktop/Visual_Recongnition/Hw3/data/train"
OUTPUT_JSON = "/home/yuyun/Desktop/Visual_Recongnition/Hw3/data/train_coco.json"


def tif_to_coco():
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"},
        ],
    }

    annotation_id = 1
    # 尋找 train 底下所有的子資料夾
    image_dirs = glob(os.path.join(DATA_DIR, "*"))

    for img_id, img_dir in enumerate(image_dirs, start=1):
        if not os.path.isdir(img_dir):
            continue

        img_name = os.path.basename(img_dir)
        img_path = os.path.join(img_dir, "image.tif")

        # 讀取圖片以獲取長寬
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        # 註冊圖片資訊 (存相對路徑比較好管理)
        coco_format["images"].append(
            {
                "id": img_id,
                "file_name": f"{img_name}/image.tif",
                "height": height,
                "width": width,
            }
        )

        # 處理 4 個類別的 Mask
        for class_id in range(1, 5):
            mask_path = os.path.join(img_dir, f"class{class_id}.tif")
            if not os.path.exists(mask_path):
                continue

            # 使用 IMREAD_UNCHANGED 確保讀取到原始的 instance ID (像素值)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # 尋找這張 mask 中有哪些 instance ID (排除 0 背景)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]

            for inst_id in instance_ids:
                # 獨立取出單一個實例的二值化遮罩
                binary_mask = (mask == inst_id).astype(np.uint8)

                # 尋找輪廓 (轉換為 COCO polygon 格式)
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:  # Polygon 至少需要 3 個點 (6個座標)
                        segmentation.append(contour)

                if not segmentation:
                    continue

                # 計算 Bounding Box
                x, y, w, h = cv2.boundingRect(binary_mask)
                area = float(np.sum(binary_mask))

                # 寫入標註
                coco_format["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    # 儲存 JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_format, f)
    print(f"轉換完成！已儲存至 {OUTPUT_JSON}")


if __name__ == "__main__":
    tif_to_coco()
