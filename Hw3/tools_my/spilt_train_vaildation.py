import json
import random

# 1. 讀取你原本全部的 JSON
with open("data/train_coco.json", "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# 2. 打亂圖片順序並切分 (8:2)
random.seed(42)  # 固定隨機種子，確保每次切分結果一樣
random.shuffle(images)

split_ratio = 0.95
split_idx = int(len(images) * split_ratio)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# 3. 根據分好的圖片 ID，把對應的標註 (annotations) 也分開
train_img_ids = {img["id"] for img in train_imgs}
val_img_ids = {img["id"] for img in val_imgs}

train_anns = [ann for ann in annotations if ann["image_id"] in train_img_ids]
val_anns = [ann for ann in annotations if ann["image_id"] in val_img_ids]

# 4. 存檔
with open("data/train_split.json", "w") as f:
    json.dump(
        {"images": train_imgs, "annotations": train_anns, "categories": categories}, f
    )

with open("data/val_split.json", "w") as f:
    json.dump(
        {"images": val_imgs, "annotations": val_anns, "categories": categories}, f
    )

print(f"切分完成！訓練集: {len(train_imgs)} 張，驗證集: {len(val_imgs)} 張")
