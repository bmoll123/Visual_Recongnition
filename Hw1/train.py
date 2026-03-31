import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm
from sklearn.metrics import classification_report

from model import (
    resnext50_se,
    resnext101_se,
)
from dataloader import get_dataloader
from utils import (
    FocalLoss,
    get_class_weights,
    get_weighted_sampler,
    save_confusion_matrix,
    generate_predictions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnext101_se",
        choices=[
            "resnet50",
            "resnet101",
            "resnext50",
            "resnext101",
            "resnext50_se",
            "resnext101_se",
        ],
        help="選擇模型架構",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="選擇 Loss Function",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step"],
        help="選擇 Learning Rate Scheduler",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="Run",
        help="Experiment name (若不指定則自動組合)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="自動讀取當前 run_name 資料夾底下的 last.pt 接續訓練",
    )
    parser.add_argument(
        "--load_weight",
        type=str,
        default=None,
        help="手動指定特定的 .pt 權重檔案路徑來接續訓練 (優先權高於 resume)",
    )
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        help="使用 WeightedRandomSampler 解決資料不平衡問題",
    )
    parser.add_argument("--test", action="store_true", help="跑test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    sampler_str = "sampler" if args.use_sampler else "nosampler"
    args.run_name = f"{args.run_name}_{args.model_name}_{args.loss}_{sampler_str}"
    print(f"\nExperiment Run Name: {args.run_name}")

    run_dir = Path("./Results") / args.run_name
    weights_dir = run_dir / "weights"
    runs_dir = run_dir / "runs"

    TRAIN_DIR = "./data/train"
    VAL_DIR = "./data/val"
    TEST_DIR = "./data/test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 100))
    elif args.model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 100))
    elif args.model_name == "resnext50":
        print("Loading Standard ResNeXt50_32x4d...")
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 100))

    elif args.model_name == "resnext101":
        print("Loading Standard ResNeXt101_32x8d...")
        model = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 100))
    elif args.model_name == "resnext50_se":
        print("Loading Handcrafted ResNeXt50 with Pretrained Weights...")
        model = resnext50_se(num_classes=100, dropout_prob=0.5, pretrained=True)
    elif args.model_name == "resnext101_se":
        print("Loading Handcrafted ResNeXt101 with Pretrained Weights...")
        model = resnext101_se(num_classes=100, dropout_prob=0.5, pretrained=True)

    model = model.to(device)

    if args.test:
        print("\n--- Test Only Mode Enabled (Validation Set)---")

        if args.load_weight and Path(args.load_weight).exists():
            best_path = Path(args.load_weight)
            run_dir = best_path.parent.parent
            args.run_name = run_dir.name
            print(f"Output directory: {run_dir}")
        else:
            print(
                f"Error: Cannot find weights at {best_path}. Please train the model first."
            )
            return

        print(f"Loading weights from: {best_path}")
        model.load_state_dict(
            torch.load(best_path, map_location=device)["model_state_dict"]
        )

        val_loader, val_dataset = get_dataloader(
            VAL_DIR, mode="val", batch_size=args.batch_size
        )

        model.eval()
        val_all_labels = []
        val_all_preds = []

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                val_all_labels.extend(labels.numpy())
                val_all_preds.extend(preds.cpu().numpy())

        class_names = (
            val_dataset.classes
            if hasattr(val_dataset, "classes")
            else [str(i) for i in range(100)]
        )

        save_confusion_matrix(
            val_all_labels,
            val_all_preds,
            class_names,
            run_dir / "best_confusion_matrix.png",
        )

        report = classification_report(
            val_all_labels,
            val_all_preds,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        val_acc = (np.array(val_all_labels) == np.array(val_all_preds)).mean()

        metrics_path = run_dir / "validation_metrics.txt"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"=== Test Mode: Validation Set Evaluation ===\n")
            f.write(f"Loaded Weight: {best_path}\n")
            f.write(f"Validation Accuracy: {val_acc:.6f}\n\n")
            f.write(f"=== Classification Report (Per-Class) ===\n")
            f.write(report)
        print(f"Validation metrics saved to: {metrics_path}")

        test_loader, test_dataset = get_dataloader(
            TEST_DIR, mode="test", batch_size=args.batch_size
        )

        if len(test_dataset) > 0:
            output_csv = run_dir / "prediction.csv"
            generate_predictions(model, device, test_loader, output_csv, args.run_name)
        else:
            print("Error: Test dataset is empty.")

        return

    weights_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))

    train_loader_original, train_dataset = get_dataloader(
        TRAIN_DIR, mode="train", batch_size=args.batch_size
    )
    val_loader, val_dataset = get_dataloader(
        VAL_DIR, mode="val", batch_size=args.batch_size
    )
    test_loader, test_dataset = get_dataloader(
        TEST_DIR, mode="test", batch_size=args.batch_size
    )

    if args.use_sampler:
        sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=train_loader_original.num_workers,
            pin_memory=train_loader_original.pin_memory,
        )
        print("Using WeightedRandomSampler for Training DataLoader.")
        class_weights = None
    else:
        train_loader = train_loader_original
        print("Using standard Training DataLoader (Shuffle=True).")
        class_weights = get_class_weights(TRAIN_DIR).to(device)

    if args.loss == "focal":
        print("Loss Function: Focal Loss")
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    elif args.loss == "ce":
        print("Loss Function: Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == "cosine":
        print("Scheduler: CosineAnnealingLR")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7
        )
    elif args.scheduler == "step":
        print("Scheduler: StepLR (Step size: 15, Gamma: 0.1)")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    start_epoch = 0
    best_acc = 0.0

    weight_to_load = None

    if args.load_weight:
        target_path = Path(args.load_weight)
        if target_path.exists():
            weight_to_load = target_path
            print(
                f"--load_weight flag detected! Will load specified weight: {weight_to_load}"
            )
        else:
            print(
                f"Warning: Specified weight file {target_path} not found. Starting from scratch."
            )

    elif args.resume:
        target_path = weights_dir / "last.pt"
        if target_path.exists():
            weight_to_load = target_path
            print(f"--resume flag detected! Will load: {weight_to_load}")
        else:
            print(
                f"Warning: --resume flag used but {target_path} not found. Starting from scratch."
            )

    if weight_to_load:
        checkpoint = torch.load(weight_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]

        print(
            f"Checkpoint loaded! Resuming from Epoch {start_epoch+1}... (Previous Best Acc: {best_acc:.4f})"
        )

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels, _ in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("LR", curr_lr, epoch)

        model.eval()
        val_running_loss = 0.0
        val_corrects = 0

        val_all_labels = []
        val_all_preds = []

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                v_loss = criterion(outputs, labels)
                val_running_loss += v_loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        scheduler.step()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(ckpt, weights_dir / "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                },
                weights_dir / "best.pt",
            )
            print(f"New best performance! best.pt saved.")

            class_names = (
                val_dataset.classes
                if hasattr(val_dataset, "classes")
                else [str(i) for i in range(100)]
            )

            save_confusion_matrix(
                val_all_labels,
                val_all_preds,
                class_names,
                run_dir / "best_confusion_matrix.png",
            )

            report = classification_report(
                val_all_labels,
                val_all_preds,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )

            metrics_path = run_dir / "best_metrics.txt"
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write(f"=== Experiment Config ===\n")
                f.write(f"Model Name: {args.model_name}\n")
                f.write(f"Loss Function: {args.loss}\n")
                f.write(f"Scheduler: {args.scheduler}\n")
                f.write(f"Total Epochs Configured: {args.epochs}\n")
                f.write(f"Initial LR: {args.lr}\n")
                f.write(f"Used Sampler: {args.use_sampler}\n")
                f.write(f"Loaded Weight (if any): {weight_to_load}\n")
                f.write(f"=========================\n\n")

                f.write(f"Best Epoch: {epoch + 1}\n")
                f.write(f"Train Loss: {epoch_loss:.6f}\n")
                f.write(f"Train Acc: {epoch_acc:.6f}\n")
                f.write(f"Val Loss: {val_loss:.6f}\n")
                f.write(f"Val Acc: {val_acc:.6f}\n\n")

                f.write(f"=== Classification Report (Per-Class) ===\n")
                f.write(report)

    print("\nTraining completed!")
    writer.close()

    output_csv = run_dir / "prediction.csv"

    if args.load_weight and not (weights_dir / "best.pt").exists():
        best_path = Path(args.load_weight)
    else:
        best_path = weights_dir / "best.pt"

    if best_path.exists():
        print(f"\nLoading best weights ({best_path}) for final prediction...")
        model.load_state_dict(
            torch.load(best_path, map_location=device)["model_state_dict"]
        )

    if len(test_dataset) > 0:
        generate_predictions(model, device, test_loader, output_csv, args.run_name)


if __name__ == "__main__":
    main()
