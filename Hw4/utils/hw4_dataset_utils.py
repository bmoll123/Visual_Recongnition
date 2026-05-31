"""
Hw4 Dataset utilities for Rain and Snow image restoration
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, RandomCrop, ToPILImage
import torch
from utils.image_utils import crop_img


class HW4TrainDataset(Dataset):
    """
    Dataset for Hw4 training with both Rain and Snow degraded images.
    Expected directory structure:
    - dataset_root/
      - degraded/
        - rain-1.png, ..., rain-1600.png
        - snow-1.png, ..., snow-1600.png
      - clean/
        - rain_clean-1.png, ..., rain_clean-1600.png
        - snow_clean-1.png, ..., snow_clean-1600.png
    """
    
    def __init__(self, dataset_root, patch_size=128):
        super(HW4TrainDataset, self).__init__()
        self.dataset_root = dataset_root
        self.patch_size = patch_size
        
        self.degraded_dir = os.path.join(dataset_root, 'degraded')
        self.clean_dir = os.path.join(dataset_root, 'clean')
        
        # Get all degraded image files
        self.degraded_files = []
        for f in sorted(os.listdir(self.degraded_dir)):
            if f.endswith('.png') or f.endswith('.jpg'):
                self.degraded_files.append(f)
        
        print(f"Found {len(self.degraded_files)} degraded images")
        
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(patch_size),
        ])
    
    def _get_clean_name(self, degraded_name):
        """
        Convert degraded image name to clean image name.
        E.g., 'rain-1.png' -> 'rain_clean-1.png'
               'snow-100.png' -> 'snow_clean-100.png'
        """
        name_parts = degraded_name.split('.')
        base_name = name_parts[0]  # e.g., 'rain-1' or 'snow-100'
        ext = name_parts[1]  # 'png'
        
        # Extract prefix (rain/snow) and number
        prefix_parts = base_name.split('-')
        prefix = prefix_parts[0]  # 'rain' or 'snow'
        number = prefix_parts[1]  # '1', '100', etc.
        
        clean_name = f"{prefix}_clean-{number}.{ext}"
        return clean_name
    
    def _crop_patch(self, img_1, img_2):
        """Crop random patch from both images at the same location."""
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        if H < self.patch_size or W < self.patch_size:
            # If image is smaller than patch size, pad it
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            if pad_h > 0 or pad_w > 0:
                img_1 = np.pad(img_1, ((pad_h//2, (pad_h+1)//2), (pad_w//2, (pad_w+1)//2), (0, 0)), mode='reflect')
                img_2 = np.pad(img_2, ((pad_h//2, (pad_h+1)//2), (pad_w//2, (pad_w+1)//2), (0, 0)), mode='reflect')
            H, W = img_1.shape[0], img_1.shape[1]
        
        import random
        ind_H = random.randint(0, max(0, H - self.patch_size))
        ind_W = random.randint(0, max(0, W - self.patch_size))
        
        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        
        return patch_1, patch_2
    
    def __getitem__(self, idx):
        degraded_name = self.degraded_files[idx]
        clean_name = self._get_clean_name(degraded_name)
        
        # Load images
        degraded_path = os.path.join(self.degraded_dir, degraded_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
        
        # Crop patches
        degraded_patch, clean_patch = self._crop_patch(degraded_img, clean_img)
        
        # Convert to tensor
        degraded_patch = self.toTensor(degraded_patch)
        clean_patch = self.toTensor(clean_patch)
        
        # Return [clean_name, de_id], degraded_patch, clean_patch
        # de_id: 0 for rain, 1 for snow
        de_id = 0 if 'rain' in degraded_name else 1
        clean_name_no_ext = degraded_name.split('.')[0]  # e.g., 'rain-1'
        
        return [clean_name_no_ext, de_id], degraded_patch, clean_patch
    
    def __len__(self):
        return len(self.degraded_files)


class HW4TestDataset(Dataset):
    """
    Dataset for Hw4 testing.
    Expected directory structure:
    - dataset_root/
      - degraded/
        - 0.png, 1.png, ..., 99.png
    """
    
    def __init__(self, dataset_root):
        super(HW4TestDataset, self).__init__()
        self.dataset_root = dataset_root
        
        self.degraded_dir = os.path.join(dataset_root, 'degraded')
        
        # Get all degraded image files
        self.degraded_files = []
        for f in sorted(os.listdir(self.degraded_dir)):
            if f.endswith('.png') or f.endswith('.jpg'):
                self.degraded_files.append(f)
        
        # Sort numerically
        self.degraded_files.sort(key=lambda x: int(x.split('.')[0]))
        
        print(f"Found {len(self.degraded_files)} test images")
        
        self.toTensor = ToTensor()
    
    def __getitem__(self, idx):
        degraded_name = self.degraded_files[idx]
        
        # Load image
        degraded_path = os.path.join(self.degraded_dir, degraded_name)
        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        
        # Convert to tensor
        degraded_tensor = self.toTensor(degraded_img)
        
        # Return [image_name], degraded_tensor
        image_name = degraded_name  # e.g., '0.png'
        
        return [image_name], degraded_tensor
    
    def __len__(self):
        return len(self.degraded_files)
