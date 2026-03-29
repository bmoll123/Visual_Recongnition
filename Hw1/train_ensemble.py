import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import zipfile
import timm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataloader import get_dataloader
from model import resnet50_cbam, EnhancedResNeXt101


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def generate_predictions(model, device, test_loader, output_csv_path, run_name):
    model.eval()
    results = []
    with torch.no_grad():
        for inputs, _, img_names in tqdm(test_loader, desc="Generating Predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for img_name, pred in zip(img_names, preds):
                results.append({'image_name': img_name, 'pred_label': pred.item()})

    df = pd.DataFrame(results)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")
    
    output_zip_path = Path(output_csv_path).parent / f"{run_name}.zip"
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv_path, arcname=Path(output_csv_path).name)
    print(f"Zipped prediction saved to: {output_zip_path}")

def generate_ensemble_predictions(model, device, test_loader, weight_paths, output_csv_path, run_name):
    """
    讀取多個權重進行推論，將機率 (Softmax) 平均後再決定最終預測類別
    """
    image_to_probs = {}
    num_models = len(weight_paths)
    
    for idx, weight_path in enumerate(weight_paths):
        print(f"\n[{idx+1}/{num_models}] Loading weight for ensemble: {weight_path}")
        if not Path(weight_path).exists():
            print(f"Warning: Weight file not found: {weight_path}. Skipping.")
            num_models -= 1
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            for inputs, _, img_names in tqdm(test_loader, desc=f"Inference"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                for i, img_name in enumerate(img_names):
                    if img_name not in image_to_probs:
                        image_to_probs[img_name] = probs[i].cpu()
                    else:
                        image_to_probs[img_name] += probs[i].cpu()

    if num_models == 0:
        print("Error: No valid models loaded for ensemble.")
        return

    print("\nCalculating final ensemble predictions...")
    results = []
    for img_name, sum_probs in image_to_probs.items():
        avg_probs = sum_probs / num_models
        pred_label = torch.argmax(avg_probs).item()
        results.append({'image_name': img_name, 'pred_label': pred_label})

    df = pd.DataFrame(results)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Ensemble predictions saved to: {output_csv_path}")
    
    output_zip_path = Path(output_csv_path).parent / f"{run_name}_ensemble.zip"
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv_path, arcname=Path(output_csv_path).name)
    print(f"Zipped ensemble prediction saved to: {output_zip_path}")

def save_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16)) 
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50_cbam', 'rexnet_100', 'rexnet_150', 'rexnet_200', 'resnext101_enhanced'], 
                        help='選擇模型架構')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_weight', type=str, default=None)
    
    # 新增：設定要保留的 Top-K 數量 (預設為 3)
    parser.add_argument('--top_k', type=int, default=3, help='保留 Validation 表現最好的前 K 個權重進行 Ensemble')
    
    parser.add_argument('--test', action='store_true', help='Skip training and only generate predictions')
    parser.add_argument('--ensemble_weights', nargs='+', default=None, help='手動輸入多個 .pt 進行 Ensemble')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    args.run_name = f"{args.run_name}_{args.model_name}_{args.loss}_{args.scheduler}_{args.lr}_{args.epochs}_{args.batch_size}"
    print(f"\nExperiment Run Name: {args.run_name}")

    run_dir = Path('./results') / args.run_name
    weights_dir = run_dir / 'weights'
    runs_dir = run_dir / 'runs'
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    TEST_DIR = './data/test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型初始化
    if args.model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 100)
        )
    elif args.model_name == 'resnet50_cbam':
        model = resnet50_cbam(num_classes=100, pretrained=True)
    elif args.model_name.startswith('rexnet'):
        model = timm.create_model(args.model_name, pretrained=True, num_classes=100)
    elif args.model_name == 'resnext101_enhanced':
        model = EnhancedResNeXt101(num_classes=100, dropout_prob=0.5)
        
    model = model.to(device)
    
    # 測試模式獨立區塊
    if args.test:
        print("\n--- Test Only Mode Enabled ---")
        test_loader, test_dataset = get_dataloader(TEST_DIR, mode='test', batch_size=args.batch_size)
        if len(test_dataset) == 0:
            print("Error: Test dataset is empty.")
            return

        output_csv = run_dir / 'prediction.csv'
        if args.ensemble_weights:
            generate_ensemble_predictions(model, device, test_loader, args.ensemble_weights, output_csv, args.run_name)
        else:
            if args.load_weight and Path(args.load_weight).exists():
                best_path = Path(args.load_weight)
            else:
                # 若無指定，嘗試抓取資料夾內名為 top_acc 最高的那一個，或 fallback 到 best.pt
                best_path = weights_dir / 'best.pt'
            
            if best_path.exists():
                model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
                generate_predictions(model, device, test_loader, output_csv, args.run_name)
            else:
                print(f"Error: Cannot find weights at {best_path}. Please train first or provide --ensemble_weights.")
        return 

    writer = SummaryWriter(log_dir=str(runs_dir))

    train_loader, train_dataset = get_dataloader(TRAIN_DIR, mode='train', batch_size=args.batch_size)
    val_loader, val_dataset = get_dataloader(VAL_DIR, mode='val', batch_size=args.batch_size)
    test_loader, test_dataset = get_dataloader(TEST_DIR, mode='test', batch_size=args.batch_size)
    
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=2.0) 
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4) # 建議稍大一點

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    start_epoch = 0
    
    # 新增：用於記錄 Top-K 個 Checkpoints 的列表 (儲存格式: (val_acc, filepath))
    top_k_checkpoints = []

    if args.load_weight or args.resume:
        # 為了簡化，如果你要 resume，這裡可以自己加載邏輯，但建議重頭 train 以確保 Top-K 計算正確
        pass

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
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('LR', curr_lr, epoch)
        
        # --- Validation stage ---
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
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        scheduler.step()

        # 儲存 last.pt 以供意外中斷時接續
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, weights_dir / 'last.pt')
        
        # ==========================================
        # Top-K 儲存機制
        # ==========================================
        is_top_k = False
        if len(top_k_checkpoints) < args.top_k:
            is_top_k = True
        elif val_acc > min(top_k_checkpoints, key=lambda x: x[0])[0]: # 比目前名單中最差的還好
            is_top_k = True

        if is_top_k:
            ckpt_path = weights_dir / f'top_acc{val_acc:.4f}_ep{epoch+1}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'acc': val_acc}, ckpt_path)
            
            top_k_checkpoints.append((val_acc.item(), ckpt_path))
            # 依照 Acc 由大到小排序
            top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
            print(f"--> Top {args.top_k} Updated! Saved: {ckpt_path.name}")
            
            # 如果名單超過 K 個，把最後一個（最差的）刪掉
            if len(top_k_checkpoints) > args.top_k:
                _, worst_path = top_k_checkpoints.pop(-1)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    
            # 只有當這是「歷史第一名」(排在 index 0) 時，才去更新混淆矩陣
            if top_k_checkpoints[0][1] == ckpt_path:
                class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else [str(i) for i in range(100)]
                save_confusion_matrix(val_all_labels, val_all_preds, class_names, run_dir / 'best_confusion_matrix.png')

    print("\nTraining completed!")
    writer.close()
    
    # ==========================================
    # 訓練結束後，自動拿存好的 Top-K 權重去 Test 跑 Ensemble
    # ==========================================
    all_ensemble_weights = [str(path) for _, path in top_k_checkpoints]
    
    if len(test_dataset) > 0 and len(all_ensemble_weights) > 0:
        print("\n" + "="*40)
        print(f"Auto-starting Final Ensemble Inference with top {len(all_ensemble_weights)} models...")
        print("="*40)
        output_csv = run_dir / 'final_ensemble_prediction.csv'
        generate_ensemble_predictions(model, device, test_loader, all_ensemble_weights, output_csv, args.run_name)

if __name__ == "__main__":
    main()