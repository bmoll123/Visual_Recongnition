import os
import sys
import json
import zipfile
import argparse
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm import tqdm

# 定義 11 個數字的專屬顏色(0:背景)
COLORS = [
    "#000000",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#ef1717",
]


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR 推論與產生預測結果")

    # 核心路徑參數
    parser.add_argument(
        "--rtdetr_dir",
        type=str,
        default="/home/cvml-3/yy/114_2/Visual_Recongition/rtdetr/rtdetr_pytorch",
        help="RT-DETR 官方程式碼根目錄",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/home/cvml-3/yy/114_2/Visual_Recongition/Hw2/data/test",
        help="測試圖片資料夾路徑",
    )

    # 模型參數 (預設為相對 rtdetr_dir 的路徑)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        help="設定檔 YAML 路徑",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="output/rtdetr_r50vd_6x_coco/checkpoint0005.pth",
        help="模型權重 (.pth) 路徑",
    )

    # 輸出與其他參數
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rtdter_ciouLoss_SE",
        help="輸出結果的資料夾 (預設為 rtdetr_dir/results)",
    )
    parser.add_argument(
        "--vis_num", type=int, default=20, help="要視覺化的圖片數量 (預設: 前 20 張)"
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.05, help="信心度門檻 (預設: 0.05)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 設定路徑 (支援絕對路徑，若是相對路徑則從 rtdetr_dir 算起)
    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(args.rtdetr_dir, args.config)
    )
    weight_path = (
        args.weights
        if os.path.isabs(args.weights)
        else os.path.join(args.rtdetr_dir, args.weights)
    )
    results_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(args.rtdetr_dir, args.output_dir)
    )

    output_json = os.path.join(results_dir, "pred.json")
    output_zip = os.path.join(results_dir, "rtdetr.zip")
    vis_dir = os.path.join(results_dir, "visualized_results")

    # 建立輸出目錄
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 確保能 import src 模組 (必須在 parse args 之後才執行)
    sys.path.insert(0, args.rtdetr_dir)
    from src.core import YAMLConfig

    # 2. 初始化模型與載入權重
    print(f"⏳ 載入設定檔: {config_path}")
    print(f"⏳ 載入權重檔: {weight_path}")
    cfg = YAMLConfig(config_path, resume=weight_path)
    model = cfg.model
    checkpoint = torch.load(weight_path, map_location="cuda")

    # 優先載入 EMA 權重以獲得最佳性能
    state_dict = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
    model.load_state_dict(state_dict)
    model.eval().cuda()

    # 3. 定義影像前處理
    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    # 按「數字大小」排序檔名，確保 image_id 順序正確 (1, 2, 3...)
    all_imgs = [f for f in os.listdir(args.test_dir) if f.endswith(".png")]
    all_imgs.sort(key=lambda x: int(os.path.splitext(x)[0]))

    results = []
    print(f"🚀 開始按順序處理 {len(all_imgs)} 張圖片...")

    with torch.no_grad():
        for i, img_name in enumerate(tqdm(all_imgs)):
            # 解析 image_id
            image_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(args.test_dir, img_name)

            # 讀取影像
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            # 推論
            img_tensor = transforms(img).unsqueeze(0).cuda()
            outputs = model(img_tensor)

            # 解析輸出
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            scores, labels = pred_logits.sigmoid().max(dim=-1)

            # 設定門檻：全量預測用參數 args.conf_thresh
            keep = scores > args.conf_thresh
            v_scores = scores[keep]
            v_labels = labels[keep]
            v_boxes = pred_boxes[keep]

            # 是否要執行視覺化
            do_vis = i < args.vis_num
            if do_vis:
                draw = ImageDraw.Draw(img)

            # 座標轉換與結果儲存
            for box, score, label in zip(v_boxes, v_scores, v_labels):
                cx, cy, bw, bh = box.cpu().tolist()

                # 計算絕對座標
                w_abs = bw * orig_w
                h_abs = bh * orig_h
                x_min = (cx - 0.5 * bw) * orig_w
                y_min = (cy - 0.5 * bh) * orig_h

                # 類別補正：0-based 轉回 1-based
                category_id = int(label.item())

                results.append(
                    {
                        "image_id": image_id,
                        "bbox": [x_min, y_min, w_abs, h_abs],
                        "score": float(score.item()),
                        "category_id": category_id,
                    }
                )

                # 視覺化邏輯 (只畫出高信心的框)
                if do_vis and score > 0.5:
                    x_max = x_min + w_abs
                    y_max = y_min + h_abs
                    color = COLORS[category_id]
                    text = f"D{category_id}:{score:.2f}"

                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                    text_y = max(0, y_min - 15)
                    t_bbox = draw.textbbox((x_min, text_y), text)
                    draw.rectangle(t_bbox, fill=color)
                    draw.text((x_min, text_y), text, fill="black")

            if do_vis:
                img.save(os.path.join(vis_dir, img_name))

    # 4. 儲存與打包
    print(f"\n📦 處理完成！共生成 {len(results)} 個預測框。")
    print(f"💾 正在寫入 {output_json} 並壓縮...")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname="pred.json")

    print(f"✅ 預測結果：{output_json}")
    print(f"✅ 壓縮檔案：{output_zip}")
    print(f"✅ 視覺化結果：{vis_dir}/ (ID 最小的前 {args.vis_num} 張)")


if __name__ == "__main__":
    main()
