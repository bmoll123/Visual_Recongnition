import json
import os
import cv2
from glob import glob

DATA_DIR = "/home/yuyun/Desktop/Visual_Recongnition/Hw3/data/"
TEST_DIR = os.path.join(DATA_DIR, "test_release")
TEST_MAPPING = os.path.join(DATA_DIR, "test_image_name_to_ids.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "test_coco_format.json")


def create_test_coco():
    # 讀取作業提供的 mapping
    with open(TEST_MAPPING, "r") as f:
        mapping = json.load(f)

    # mapping 可能是一個 list of dicts，例如 [{"file_name": "...", "id": 1, ...}]

    coco_format = {
        "images": [],
        "annotations": [],  # 測試集沒有標註，保持空陣列
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"},
        ],
    }

    for item in mapping:
        # 將作業給的格式轉換成 COCO 預期的 images 格式
        coco_format["images"].append(
            {
                "id": item["id"],
                "file_name": item["file_name"],  # 注意這裡的路徑是否相對於 data_prefix
                "height": item.get(
                    "height", 0
                ),  # 如果 mapping 裡沒有，之後要用 cv2 讀圖補上
                "width": item.get("width", 0),
            }
        )

    # 如果作業的 json 沒有包含 height/width，你需要取消下面這段的註解來動態讀取
    # for img_info in coco_format["images"]:
    #     if img_info["height"] == 0:
    #         img_path = os.path.join(TEST_DIR, img_info["file_name"])
    #         img = cv2.imread(img_path)
    #         if img is not None:
    #             img_info["height"], img_info["width"] = img.shape[:2]

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_format, f)
    print(f"測試集偽標註建立完成: {OUTPUT_JSON}")


if __name__ == "__main__":
    create_test_coco()
