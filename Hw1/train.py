import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# 從我們獨立出來的檔案中引入 get_dataloaders 函式
from dataset import get_dataloaders

def main():
    # ==========================================
    # 1. 載入資料 (呼叫 dataset.py 中的函式)
    # ==========================================
    CSV_PATH = './prediction.csv'
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    print("開始準備資料...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        csv_file=CSV_PATH, 
        train_dir=TRAIN_DIR, 
        val_dir=VAL_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=4 
    )

    # ==========================================
    # 2. 建立 ResNet50 模型
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用設備: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 修改最後一層 FC 層，適應 100 個類別
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ==========================================
    # 3. 訓練迴圈
    # ==========================================
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        # --- 訓練階段 ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    print("訓練完成！")

if __name__ == "__main__":
    main()