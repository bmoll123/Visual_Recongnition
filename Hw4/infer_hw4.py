"""
Inference script for Hw4 Image Restoration
Generates pred.npz file with restored images
"""
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning.pytorch as pl

from utils.hw4_dataset_utils import HW4TestDataset
from net.model import PromptIR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        """Training step (not used in inference)"""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer (not used in inference)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        return [optimizer]


def tensor_to_image(tensor):
    """
    Convert tensor (C, H, W) with values in [0, 1] to numpy array (C, H, W) with uint8 values [0, 255]
    """
    # Ensure tensor is on CPU and detach
    tensor = tensor.cpu().detach()
    
    # Clip values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    image = tensor.numpy()
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Inference for Hw4 Image Restoration')
    
    # Data arguments
    parser.add_argument('--test_root', type=str,
                        default='dataset/hw4_realse_dataset/test',
                        help='Root directory of test dataset')
    parser.add_argument('--ckpt_path', type=str,
                        default='train_ckpt/epoch=149-train_loss=0.0000.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str,
                        default='pred.npz',
                        help='Output path for pred.npz')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device index')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Inference for Hw4 Image Restoration")
    print("=" * 50)
    print(f"Test root: {args.test_root}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output: {args.output_path}")
    print("=" * 50)
    
    # Set device
    torch.cuda.set_device(args.cuda)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        print(f"Please check the checkpoint path and make sure training is completed.")
        return
    
    model = PromptIRModel.load_from_checkpoint(args.ckpt_path)
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    print(f"Loading test dataset from {args.test_root}...")
    testset = HW4TestDataset(args.test_root)
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Run inference
    print("Running inference...")
    pred_dict = {}
    
    with torch.no_grad():
        for ([image_names], degraded_imgs) in tqdm(testloader, desc="Inference"):
            degraded_imgs = degraded_imgs.to(device)
            
            # Forward pass
            restored_imgs = model.net(degraded_imgs)
            
            # Process each image in batch
            for i, image_name in enumerate(image_names):
                restored_img = restored_imgs[i]  # (C, H, W)
                
                # Convert to numpy uint8
                image_array = tensor_to_image(restored_img)
                
                # Store in dictionary
                pred_dict[image_name] = image_array
                
                if (len(pred_dict) - 1) % 10 == 0:
                    print(f"Processed {len(pred_dict)} images")
    
    # Save to npz
    print(f"Saving results to {args.output_path}...")
    np.savez_compressed(args.output_path, **pred_dict)
    
    print(f"Done! Saved {len(pred_dict)} images to {args.output_path}")
    print(f"File size: {os.path.getsize(args.output_path) / 1e6:.2f} MB")


if __name__ == '__main__':
    main()
