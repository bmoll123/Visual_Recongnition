import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, img_dir, mode='train', transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode
        self.data = []

        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}

        if mode == 'test':
            # Test mode: Scan all files in the root directory
            for img_path in self.img_dir.iterdir():
                if img_path.suffix in extensions:
                    self.data.append({
                        'image_path': img_path,
                        'label': -1,
                        'img_name': img_path.stem
                    })
        else:
            # Train/Val mode: Scan subdirectories (class folder structure)
            for class_dir in self.img_dir.iterdir():
                if class_dir.is_dir():
                    try:
                        label = int(class_dir.name)
                        for img_path in class_dir.iterdir():
                            if img_path.suffix in extensions:
                                self.data.append({
                                    'image_path': img_path,
                                    'label': label,
                                    'img_name': img_path.stem
                                })
                    except ValueError:
                        # Skip folders that are not named as integers
                        continue

        self.df = pd.DataFrame(self.data)
        if len(self.df) == 0:
            print(f"No images found in {img_dir}.")
        else:
            print(f"{mode.capitalize()} set loaded successfully. Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        img_name = row['img_name']
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_name


def get_dataloader(img_dir, mode='train', batch_size=32, num_workers=4):
    """Create a DataLoader based on the specified mode"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if mode == 'train':
        transform = transforms.Compose([
            # 1. 隨機調整大小並裁剪：這比固定 Resize + Crop 更有利於學習不同尺度
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # 2. 隨機水平翻轉：自然影像中左右通常是對稱的
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. 隨機旋轉：小角度旋轉增加穩定性
            transforms.RandomRotation(degrees=15),
            # 4. 色彩抖動：隨機調整亮度、對比、飽和度，模擬不同相機與光照
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 5. 隨機灰階化：強迫模型學習形狀而非過度依賴顏色
            # transforms.RandomGrayscale(p=0.1),
            # 6. 轉為張量並標準化
            transforms.ToTensor(),
            normalize,
            # 7. 隨機擦除：遮蓋部分區域，防止模型只看物體的某個特徵（須放在 ToTensor 之後）
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        shuffle = False

    dataset = ImageDataset(img_dir=img_dir, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, dataset