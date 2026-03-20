import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # 讀取 CSV
        df = pd.read_csv(csv_file)
        
        # 自動過濾：只保留確實存在於 img_dir 資料夾中的圖片標籤
        valid_data = []
        for _, row in df.iterrows():
            img_name = str(row['image_name'])
            # 假設圖片副檔名是 .jpg，請依據實際情況修改
            img_path = self.img_dir / f"{img_name}.jpg" 
            
            if img_path.exists():
                valid_data.append({'image_name': img_path, 'label': row['pred_label']})
                
        self.data = pd.DataFrame(valid_data)
        print(f"從 {img_dir} 成功載入 {len(self.data)} 筆資料")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_name']
        label = self.data.iloc[idx]['label']
        
        # 讀取圖片並轉換為 RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_dataloaders(csv_file, train_dir, val_dir, batch_size=32, num_workers=4):
    """
    建立並回傳 train 與 val 的 DataLoader 及 Dataset
    """
    # ImageNet 標準化參數
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # 建立 Dataset
    train_dataset = ImageDataset(csv_file=csv_file, img_dir=train_dir, transform=train_transforms)
    val_dataset = ImageDataset(csv_file=csv_file, img_dir=val_dir, transform=val_transforms)

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset, val_dataset