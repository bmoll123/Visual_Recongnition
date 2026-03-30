import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, img_dir, mode="train", transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode
        self.data = []

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}

        if mode == "test":
            # Test mode: Scan all files in the root directory
            for img_path in self.img_dir.iterdir():
                if img_path.suffix in extensions:
                    self.data.append(
                        {"image_path": img_path, "label": -1, "img_name": img_path.stem}
                    )
        else:
            # Train/Val mode: Scan subdirectories (class folder structure)
            for class_dir in self.img_dir.iterdir():
                if class_dir.is_dir():
                    try:
                        label = int(class_dir.name)
                        for img_path in class_dir.iterdir():
                            if img_path.suffix in extensions:
                                self.data.append(
                                    {
                                        "image_path": img_path,
                                        "label": label,
                                        "img_name": img_path.stem,
                                    }
                                )
                    except ValueError:
                        # Skip folders that are not named as integers
                        continue

        self.df = pd.DataFrame(self.data)
        if len(self.df) == 0:
            print(f"No images found in {img_dir}.")
        else:
            print(
                f"{mode.capitalize()} set loaded successfully. Total samples: {len(self.df)}"
            )

        self.targets = self.df["label"].tolist() if not self.df.empty else []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = row["label"]
        img_name = row["img_name"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_name


def get_dataloader(img_dir, mode="train", batch_size=32, num_workers=4):
    """Create a DataLoader based on the specified mode"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        transform = transforms.Compose(
            [
                # 1. 【局部特徵放大】：植物辨識常看細節（如花蕊、葉緣）。
                # 允許裁切到 40%，強迫模型不能只看整體輪廓，要學會看局部紋理。
                transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                # 2. 【多維度翻轉】：植物（特別是俯拍的葉片或花朵）沒有絕對的「上下」之分！
                # 加入垂直翻轉，這能瞬間讓你的資料量在模型眼中翻倍。
                transforms.RandomHorizontalFlip(p=0.8),
                transforms.RandomVerticalFlip(p=0.8),
                # 3. 【大角度旋轉】：植物的生長方向千奇百怪，所以旋轉角度可以大膽開到 45 度甚至 90 度。
                transforms.RandomRotation(degrees=45),
                # 4. 【謹慎的色彩微調 (關鍵!)】：
                # 亮度、對比、飽和度可以調，模擬不同天氣的日照。
                # ⚠️ 但是 Hue (色相) 必須設得非常小 (0.05 甚至 0)！
                # 因為如果把「粉紅色的花」色相偏移變成「橘色的花」，標籤就錯了！
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0
                ),
                transforms.ToTensor(),
                normalize,
                # 5. 【模擬遮擋】：在野外，葉片常被昆蟲咬破、被泥土弄髒、或被其他葉子擋住。
                # 隨機擦除能完美模擬這點，強迫模型看其他部位。
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ]
        )
        shuffle = True
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        shuffle = False

    dataset = ImageDataset(img_dir=img_dir, mode=mode, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader, dataset
