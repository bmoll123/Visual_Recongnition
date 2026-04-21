import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import zipfile

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from PIL import ImageDraw, Image

from util.box_ops import box_cxcywh_to_xyxy


def run_blind_test(model, postprocessors, device, output_dir, args, suffix="best"):
    model.eval()
    import torchvision.transforms as T

    test_path = "/home/yuyun/Desktop/Visual_Recongnition/Hw2/data/test"
    results = []

    # 🌟 修正 2：按照數字大小排序檔名 (確保 image_id 順序為 1, 2, 3...)
    image_files = [f for f in os.listdir(test_path) if f.endswith(".png")]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    print(f"🚀 Running blind test on {len(image_files)} images...")
    for img_name in tqdm(image_files):
        # 從檔名提取正確的 image_id
        image_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(test_path, img_name)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)

        target_sizes = torch.tensor([[h, w]]).to(device)
        processed = postprocessors["bbox"](outputs, target_sizes)[0]

        scores = processed["scores"].cpu().numpy()
        labels = processed["labels"].cpu().numpy()
        boxes = processed["boxes"].cpu().numpy()

        for s, l, b in zip(scores, labels, boxes):
            x1, y1, x2, y2 = b
            # 🌟 修正 3：按照要求格式 [x, y, w, h] 儲存，且 category_id +1
            results.append(
                {
                    "image_id": image_id,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),  # width
                        float(y2 - y1),  # height
                    ],
                    "score": float(s),
                    "category_id": int(l) + 1,  # 數字 0 對應 category 1
                }
            )

    # 🌟 修正 4：儲存前確保按照 image_id 排序
    results.sort(key=lambda x: x["image_id"])

    pred_path = output_dir / "pred.json"
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=4)

    zip_path = output_dir / f"{args.project_name}_{suffix}.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(pred_path, arcname="pred.json")

    print(f"✅ Saved zip: {zip_path}")


def visualize_validation(model, postprocessors, data_loader, device, output_dir):
    model.eval()

    # ImageNet 去歸一化參數
    checkpoint_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    checkpoint_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    save_dir = output_dir / "validation_visualize"
    save_dir.mkdir(exist_ok=True)

    count = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        with torch.no_grad():
            outputs = model(samples)

        target_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        results = postprocessors["bbox"](outputs, target_sizes)

        # 取得 mask 資訊來解決 Padding (棕色區塊) 問題
        masks = samples.mask

        for i in range(len(results)):
            if count >= 10:
                return

            # 🌟 修正 5：利用 Mask 裁切掉補零區域，解決下方的棕色塊
            h_real = (~masks[i]).sum(0).max().item()
            w_real = (~masks[i]).sum(1).max().item()

            # 只取有效區域並去歸一化
            img_tensor = samples.tensors[i][:, :h_real, :w_real].cpu()
            img_tensor = img_tensor * checkpoint_std + checkpoint_mean
            img_tensor = torch.clamp(img_tensor, 0, 1)

            img = F.to_pil_image(img_tensor)
            img_w, img_h = img.size
            draw = ImageDraw.Draw(img)

            # 🔵 Prediction (紅色)
            scores = results[i]["scores"].cpu()
            labels = results[i]["labels"].cpu()
            boxes = results[i]["boxes"].cpu()
            for s, l, b in zip(scores, labels, boxes):
                if s < 0.5:
                    continue
                x1, y1, x2, y2 = b.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"P:{l.item()+1} {s:.2f}", fill="red")

            # 🟢 Ground Truth (綠色)
            gt_boxes = targets[i]["boxes"].cpu()
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
            gt_boxes = gt_boxes * scale_fct
            gt_labels = targets[i]["labels"].cpu()

            for l, b in zip(gt_labels, gt_boxes):
                gx1, gy1, gx2, gy2 = b.tolist()
                draw.rectangle([gx1, gy1, gx2, gy2], outline="green", width=2)
                draw.text((gx1, gy1), f"G:{l.item() + 1}", fill="green")

            img.save(save_dir / f"vis_{count}.png")
            count += 1


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)

    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--project_name", default="default_project", type=str)
    parser.add_argument(
        "--warmup_steps", default=2000, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--eval_weights",
        default=None,
        type=str,
        help="path to specific weights for evaluation",
    )

    parser.add_argument("--clip_max_norm", default=0.1, type=float)

    # Model parameters
    parser.add_argument("--frozen_weights", type=str, default=None)

    # Backbone
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
    )

    # Transformer
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=100, type=int)
    parser.add_argument("--pre_norm", action="store_true")

    # Segmentation
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_false")

    # Matcher
    parser.add_argument("--set_cost_class", default=1, type=float)
    parser.add_argument("--set_cost_bbox", default=5, type=float)
    parser.add_argument("--set_cost_giou", default=2, type=float)

    # Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--eos_coef", default=0.1, type=float)

    # dataset
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    return parser


def main(args):
    utils.init_distributed_mode(args)

    # 設定輸出目錄
    args.output_dir = f"results/{args.project_name}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 建立模型與後處理器
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    # 建立 Data Loader
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    base_ds = get_coco_api_from_dataset(dataset_val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # ==========================================================
    # 🌟 新增：初始化 AMP GradScaler
    # ==========================================================
    scaler = torch.cuda.amp.GradScaler()

    # ==========================================================
    # 🌟 載入 Resume 權重 (接續訓練)
    # ==========================================================
    if args.resume:
        if os.path.exists(args.resume):
            print(f"▶️ Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])

            # 確保訓練狀態也能接續 (包含 Epoch、Optimizer 與 Scheduler)
            if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1

                # 🌟 新增：載入 scaler 狀態
                if "scaler" in checkpoint and checkpoint["scaler"] is not None:
                    scaler.load_state_dict(checkpoint["scaler"])

                print(f"📈 Resuming training from Epoch {args.start_epoch}")
        else:
            print(
                f"⚠️ Warning: Checkpoint path '{args.resume}' not found. Starting from scratch."
            )

    if not args.eval:
        # --- 【訓練模式】 ---
        print(f"🚀 Starting training for project: {args.project_name}")
        best_map = 0
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                args.clip_max_norm,
                scaler=scaler,
                warmup_steps=args.warmup_steps,
            )
            lr_scheduler.step()

            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
            )

            # TensorBoard 記錄
            for k, v in train_stats.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"train/{k}", v, epoch)
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    writer.add_scalar(f"train/{k}", v.item(), epoch)
            for k, v in test_stats.items():
                # 🌟 特殊處理 COCO 指標列表
                if k == "coco_eval_bbox":
                    writer.add_scalar("val/mAP", v[0], epoch)  # AP @ [0.50:0.95]
                    writer.add_scalar("val/mAP_50", v[1], epoch)  # AP @ 0.50
                    writer.add_scalar("val/mAP_75", v[2], epoch)  # AP @ 0.75
                    writer.add_scalar(
                        "val/mAP_small", v[3], epoch
                    )  # AP for small objects
                    writer.add_scalar(
                        "val/mAP_medium", v[4], epoch
                    )  # AP for medium objects
                    writer.add_scalar(
                        "val/mAP_large", v[5], epoch
                    )  # AP for large objects

                # 原有的純量記錄
                elif isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)

            # 建立要儲存的完整狀態字典 (包含 epoch 與 optimizer 以便未來 Resume)
            checkpoint_state = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "scaler": scaler.state_dict(),  # 🌟 新增：保存 scaler 狀態
            }

            # 🌟 每個 Epoch 存檔 latest.pth (確保系統中斷時不會遺失太多進度)
            torch.save(checkpoint_state, output_dir / "checkpoint_last.pth")

            # Save best mAP
            if coco_evaluator and "bbox" in coco_evaluator.coco_eval:
                current_map = coco_evaluator.coco_eval["bbox"].stats[0]
                if current_map > best_map:
                    best_map = current_map
                    print(
                        f"🔥 New Best mAP: {best_map:.4f}, saving checkpoint_best.pth"
                    )
                    torch.save(checkpoint_state, output_dir / "checkpoint_best.pth")

        print("✅ Training finished.")

    # ==========================================================
    # 🌟 純測試邏輯 (不論是剛練完還是直接 eval 都會跑)
    # ==========================================================

    print(f"\n🔍 Starting Evaluation for project: {args.project_name}")

    # 1. 測試 BEST 權重
    best_path = output_dir / "checkpoint_best.pth"
    if best_path.exists():
        print(f"▶️ Loading BEST weights from {best_path}")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])

        print("Running Validation Visualization (Best)...")
        visualize_validation(model, postprocessors, data_loader_val, device, output_dir)

        print("Running Blind Test (Best)...")
        run_blind_test(model, postprocessors, device, output_dir, args, suffix="best")
    else:
        print(f"⚠️ Warning: {best_path} not found, skipping Best evaluation.")

    # 2. 測試 LAST 權重
    last_path = output_dir / "checkpoint_last.pth"
    if last_path.exists():
        print(f"▶️ Loading LAST weights from {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])

        print("Running Blind Test (Last)...")
        run_blind_test(model, postprocessors, device, output_dir, args, suffix="last")
    else:
        print(f"⚠️ Warning: {last_path} not found, skipping Last evaluation.")

    print("🏁 All tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
