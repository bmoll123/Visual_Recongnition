"""
Training script for Hw4 Image Restoration using PromptIR
"""
import subprocess
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.hw4_dataset_utils import HW4TrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(self, learning_rate=2e-4):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='Train PromptIR for Image Restoration')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                        default='dataset/hw4_realse_dataset/train',
                        help='Root directory of training dataset')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Logging and checkpoint arguments
    parser.add_argument('--ckpt_dir', type=str, default='train_ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='promptir-hw4',
                        help='Weights & Biases project name')
    parser.add_argument('--enable_progress_bar', action='store_true', default=True,
                        help='Enable progress bar')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Training PromptIR for Image Restoration (Hw4)")
    print("=" * 50)
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("=" * 50)
    
    # Create output directories
    subprocess.run(['mkdir', '-p', args.ckpt_dir], check=False)
    
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Create dataset
    print(f"Loading dataset from {args.data_root}...")
    trainset = HW4TrainDataset(args.data_root, patch_size=args.patch_size)
    print(f"Dataset size: {len(trainset)}")
    
    # Create dataloader
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    # Create model
    model = PromptIRModel(learning_rate=args.lr)
    
    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project, name="PromptIR-Hw4")
    else:
        logger = TensorBoardLogger(save_dir="logs/")
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1,  # Save all checkpoints
        filename='{epoch:03d}-{train_loss:.4f}'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model=model, train_dataloaders=trainloader)
    
    print("Training completed!")
    print(f"Checkpoints saved to: {args.ckpt_dir}")


if __name__ == '__main__':
    main()
