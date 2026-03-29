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
from sklearn.metrics import confusion_matrix, classification_report

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

def get_class_weights(train_dir):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    
    class_counts = []
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        class_counts.append(num_files)
        
    class_counts = np.array(class_counts)
    total_samples = np.sum(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

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

# === [新增]：將建立模型的邏輯獨立出來，方便 Ensemble 呼叫 ===
def build_model(model_name, device):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(num_ftrs, 100)
        )
    elif model_name == 'resnet50_cbam':
        model = resnet50_cbam(num_classes=100, pretrained=False)
    elif model_name.startswith('rexnet'):
        print(f"Loading {model_name} from timm...")
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
    elif model_name == 'resnext101_enhanced':
        model = EnhancedResNeXt101(num_classes=100, dropout_prob=0.5)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model.to(device)

# === [修改]：讓 generate_predictions 支援多個模型傳入 ===
def generate_predictions(models, device, test_loader, output_csv_path, run_name):
    # 如果傳入的是單一模型，把它轉成 List 方便統一處理
    if not isinstance(models, list):
        models = [models]
        
    for m in models:
        m.eval()
        
    results = []
    
    with torch.no_grad():
        for inputs, _, img_names in tqdm(test_loader, desc="Generating Predictions"):
            inputs = inputs.to(device)
            
            # 收集所有模型的輸出
            all_outputs = [m(inputs) for m in models]
            
            # 使用 Soft Voting：把所有輸出的 Logits 取平均
            avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
            
            # 從平均後的 Logits 找出預測類別
            _, preds = torch.max(avg_outputs, 1)
            
            for img_name, pred in zip(img_names, preds):
                results.append({'image_name': img_name, 'pred_label': pred.item()})

    df = pd.DataFrame(results)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to: {output_csv_path}")
    
    output_zip_path = Path(output_csv_path).parent / f"{run_name}.zip"
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv_path, arcname=Path(output_csv_path).name)
    print(f"Zipped prediction saved to: {output_zip_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50_cbam', 'rexnet_100', 'rexnet_150', 'rexnet_200', 'resnext101_enhanced'], 
                        help='選擇模型架構')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'], help='選擇 Loss Function')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='選擇 Learning Rate Scheduler')
    parser.add_argument('--run_name', type=str, default=None, help='Experiment name (若不指定則自動組合)')
    parser.add_argument('--resume', action='store_true', help='自動讀取當前 run_name 資料夾底下的 last.pt 接續訓練')
    parser.add_argument('--load_weight', type=str, default=None, help='手動指定特定的 .pt 權重檔案路徑來接續訓練 (優先權高於 resume)')
    
    parser.add_argument('--test', action='store_true', help='Skip training and only generate predictions using best.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # === [新增]：Ensemble 專用的參數 ===
    parser.add_argument('--ensemble', action='store_true', help='啟動多模型 Ensemble 推論模式')
    parser.add_argument('--ensemble_models', type=str, nargs='+', help='Ensemble 所使用的模型架構列表')
    parser.add_argument('--ensemble_weights', type=str, nargs='+', help='Ensemble 對應的權重檔案列表')

    args = parser.parse_args()

    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    TEST_DIR = './data/test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================
    # Ensemble Mode
    # ==========================
    if args.ensemble:
        print("\n--- Ensemble Inference Mode Enabled ---")
        if len(args.ensemble_models) != len(args.ensemble_weights):
            print("Error: The number of --ensemble_models must match the number of --ensemble_weights.")
            return
            
        # 建立 Ensemble 輸出資料夾
        run_dir = Path('./results/ensemble_output')
        run_dir.mkdir(parents=True, exist_ok=True)
        args.run_name = "ensemble_predictions"
        
        # 載入所有模型
        ensemble_model_instances = []
        for m_name, w_path in zip(args.ensemble_models, args.ensemble_weights):
            print(f"Loading {m_name} with weight: {w_path}...")
            m = build_model(m_name, device)
            m.load_state_dict(torch.load(w_path, map_location=device)['model_state_dict'])
            ensemble_model_instances.append(m)
            
        val_loader, val_dataset = get_dataloader(VAL_DIR, mode='val', batch_size=args.batch_size)
        
        print("\nEvaluating on Validation Set with Ensemble (Soft Voting)...")
        val_all_labels = []
        val_all_preds = []
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc="Ensemble Validation"):
                inputs = inputs.to(device)
                # 對每個模型收集 Logits，然後取平均
                all_outputs = [m(inputs) for m in ensemble_model_instances]
                avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
                _, preds = torch.max(avg_outputs, 1)
                
                val_all_labels.extend(labels.numpy())
                val_all_preds.extend(preds.cpu().numpy())
                
        class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else [str(i) for i in range(100)]
        
        save_confusion_matrix(val_all_labels, val_all_preds, class_names, run_dir / 'ensemble_confusion_matrix.png')
        report = classification_report(val_all_labels, val_all_preds, target_names=class_names, digits=4, zero_division=0)
        val_acc = (np.array(val_all_labels) == np.array(val_all_preds)).mean()
        
        metrics_path = run_dir / 'ensemble_metrics.txt'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Ensemble Mode: Validation Set Evaluation ===\n")
            f.write(f"Models Used: {args.ensemble_models}\n")
            f.write(f"Validation Accuracy: {val_acc:.6f}\n\n")
            f.write(f"=== Classification Report (Per-Class) ===\n")
            f.write(report)
        print(f"Ensemble metrics saved to: {metrics_path}")

        test_loader, test_dataset = get_dataloader(TEST_DIR, mode='test', batch_size=args.batch_size)
        if len(test_dataset) > 0:
            output_csv = run_dir / 'prediction.csv'
            generate_predictions(ensemble_model_instances, device, test_loader, output_csv, args.run_name)
        else:
            print("Error: Test dataset is empty.")
            
        return

    # ==========================
    # 常規模式 (Training 或 Single Test)
    # ==========================
    args.run_name = f"{args.run_name}_{args.model_name}_{args.loss}_{args.scheduler}_{args.lr}_{args.epochs}_{args.batch_size}"
    print(f"\nExperiment Run Name: {args.run_name}")

    run_dir = Path('./results') / args.run_name
    weights_dir = run_dir / 'weights'
    runs_dir = run_dir / 'runs'
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 建立模型
    model = build_model(args.model_name, device)
    
    # Test Only Mode (Single Model)
    if args.test:
        print("\n--- Test Only Mode Enabled ---")
        
        if args.load_weight and Path(args.load_weight).exists():
            best_path = Path(args.load_weight)
            run_dir = best_path.parent.parent
            args.run_name = run_dir.name
            print(f"Output directory automatically set to: {run_dir}")
        else:
            best_path = weights_dir / 'best.pt'
        
        if not best_path.exists():
            print(f"Error: Cannot find weights at {best_path}. Please train the model first.")
            return

        print(f"Loading weights from: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        
        val_loader, val_dataset = get_dataloader(VAL_DIR, mode='val', batch_size=args.batch_size)
        
        print("\nEvaluating on Validation Set to generate metrics...")
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
                
        class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else [str(i) for i in range(100)]
        
        save_confusion_matrix(val_all_labels, val_all_preds, class_names, run_dir / 'best_confusion_matrix.png')
        report = classification_report(val_all_labels, val_all_preds, target_names=class_names, digits=4, zero_division=0)
        val_acc = (np.array(val_all_labels) == np.array(val_all_preds)).mean()
        
        metrics_path = run_dir / 'best_metrics.txt'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Test Mode: Validation Set Evaluation ===\n")
            f.write(f"Loaded Weight: {best_path}\n")
            f.write(f"Validation Accuracy: {val_acc:.6f}\n\n")
            f.write(f"=== Classification Report (Per-Class) ===\n")
            f.write(report)
        print(f"Validation metrics saved to: {metrics_path}")

        test_loader, test_dataset = get_dataloader(TEST_DIR, mode='test', batch_size=args.batch_size)
        if len(test_dataset) > 0:
            output_csv = run_dir / 'prediction.csv'
            generate_predictions(model, device, test_loader, output_csv, args.run_name)
        else:
            print("Error: Test dataset is empty.")
            
        return

    # Training Mode
    writer = SummaryWriter(log_dir=str(runs_dir))

    train_loader, train_dataset = get_dataloader(TRAIN_DIR, mode='train', batch_size=args.batch_size)
    val_loader, val_dataset = get_dataloader(VAL_DIR, mode='val', batch_size=args.batch_size)
    test_loader, test_dataset = get_dataloader(TEST_DIR, mode='test', batch_size=args.batch_size)
    
    print("Calculating class weights to handle imbalanced dataset...")
    class_weights = get_class_weights(TRAIN_DIR).to(device)
    
    if args.loss == 'focal':
        print("Loss Function: Focal Loss")
        criterion = FocalLoss(gamma=2.0, alpha=class_weights) 
    elif args.loss == 'ce':
        print("Loss Function: Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cosine':
        print("Scheduler: CosineAnnealingLR")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        print("Scheduler: StepLR (Step size: 15, Gamma: 0.1)")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    start_epoch = 0
    best_acc = 0.0

    weight_to_load = None
    
    if args.load_weight:
        target_path = Path(args.load_weight)
        if target_path.exists():
            weight_to_load = target_path
            print(f"--load_weight flag detected! Will load specified weight: {weight_to_load}")
        else:
            print(f"Warning: Specified weight file {target_path} not found. Starting from scratch.")
            
    elif args.resume:
        target_path = weights_dir / 'last.pt'
        if target_path.exists():
            weight_to_load = target_path
            print(f"--resume flag detected! Will load: {weight_to_load}")
        else:
            print(f"Warning: --resume flag used but {target_path} not found. Starting from scratch.")

    if weight_to_load:
        checkpoint = torch.load(weight_to_load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
            
        print(f"Checkpoint loaded! Resuming from Epoch {start_epoch+1}... (Previous Best Acc: {best_acc:.4f})")

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

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }
        torch.save(ckpt, weights_dir / 'last.pt')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, weights_dir / 'best.pt')
            print(f"New best performance! best.pt saved.")
            
            class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else [str(i) for i in range(100)]
            
            save_confusion_matrix(val_all_labels, val_all_preds, class_names, run_dir / 'best_confusion_matrix.png')
            
            report = classification_report(val_all_labels, val_all_preds, target_names=class_names, digits=4, zero_division=0)
            
            metrics_path = run_dir / 'best_metrics.txt'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Experiment Config ===\n")
                f.write(f"Model Name: {args.model_name}\n")
                f.write(f"Loss Function: {args.loss}\n")
                f.write(f"Scheduler: {args.scheduler}\n")
                f.write(f"Total Epochs Configured: {args.epochs}\n")
                f.write(f"Initial LR: {args.lr}\n")
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
    
    output_csv = run_dir / 'prediction.csv'
    
    if args.load_weight and not (weights_dir / 'best.pt').exists():
         best_path = Path(args.load_weight)
    else:
         best_path = weights_dir / 'best.pt'
         
    if best_path.exists():
        print(f"\nLoading best weights ({best_path}) for final prediction...")
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    
    if len(test_dataset) > 0:
        generate_predictions(model, device, test_loader, output_csv, args.run_name)

if __name__ == "__main__":
    main()