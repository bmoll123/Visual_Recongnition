"""
完整实验管理训练脚本
所有输出（checkpoint、logs、配置）都在指定的实验目录下
"""
import subprocess
import argparse
import os
import json
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.hw4_dataset_utils import HW4TrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import AverageMeter, compute_psnr_ssim
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


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
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        
        temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_psnr", temp_psnr, prog_bar=True, on_epoch=True)
        self.log("val_ssim", temp_ssim, prog_bar=True, on_epoch=True)
        
        return {
            "loss": loss,
            "psnr": temp_psnr,
            "ssim": temp_ssim
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        return [optimizer], [scheduler]


def find_latest_checkpoint(ckpt_dir):
    """找到最新的checkpoint"""
    if not os.path.exists(ckpt_dir):
        return None
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and f != 'last.ckpt']
    if not ckpt_files:
        return None
    
    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
    return os.path.join(ckpt_dir, ckpt_files[0])


def save_config(config_dict, config_path):
    """保存实验配置"""
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train PromptIR with complete experiment management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 第一次训练，自动创建exp_001目录
  python train_hw4_exp.py --exp_dir exp_001 --epochs 150 --batch_size 4
  
  # 继续之前的实验
  python train_hw4_exp.py --exp_dir exp_001 --auto_resume --epochs 150
  
  # 自定义实验目录
  python train_hw4_exp.py --exp_dir my_experiments/rain_snow_v2 --epochs 200
        """
    )
    
    # 实验设置
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory (all outputs go here)')
    parser.add_argument('--exp_name', type=str, default='promptir_hw4',
                        help='Experiment name for logging')
    
    # 数据设置
    parser.add_argument('--data_root', type=str, 
                        default='dataset/hw4_realse_dataset/train',
                        help='Root directory of training dataset')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set ratio')
    
    # 训练设置
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Checkpoint设置
    parser.add_argument('--monitor_metric', type=str, default='val_psnr',
                        choices=['val_psnr', 'val_loss', 'val_ssim'],
                        help='Metric to monitor for saving best model')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Number of top models to save')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience (0=disabled)')
    
    # Resume设置
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto resume from latest checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint')
    
    args = parser.parse_args()
    
    # 创建实验目录
    exp_dir = args.exp_dir
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    logs_dir = os.path.join(exp_dir, 'logs')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("=" * 80)
    print("Training PromptIR with Experiment Management")
    print("=" * 80)
    print(f"Experiment directory: {os.path.abspath(exp_dir)}")
    print(f"  ├── Checkpoints: {os.path.abspath(ckpt_dir)}")
    print(f"  └── Logs:        {os.path.abspath(logs_dir)}")
    print("=" * 80)
    
    # 保存配置
    config = {
        'timestamp': datetime.now().isoformat(),
        'exp_name': args.exp_name,
        'data': {
            'root': args.data_root,
            'patch_size': args.patch_size,
            'val_split': args.val_split,
        },
        'training': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'num_gpus': args.num_gpus,
            'num_workers': args.num_workers,
        },
        'checkpoint': {
            'monitor_metric': args.monitor_metric,
            'save_top_k': args.save_top_k,
            'early_stopping_patience': args.early_stopping_patience,
        }
    }
    
    config_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_path)
    
    print(f"\nConfiguration:")
    print(f"  Data: {args.data_root}")
    print(f"  Batch size: {args.batch_size} | Patch size: {args.patch_size}")
    print(f"  Validation split: {args.val_split*100:.0f}%")
    print(f"  Epochs: {args.epochs} | Learning rate: {args.lr}")
    print(f"  Monitor metric: {args.monitor_metric}")
    print(f"  Early stopping patience: {args.early_stopping_patience if args.early_stopping_patience > 0 else 'Disabled'}")
    print("=" * 80)
    
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 创建数据集
    print(f"\nLoading dataset from {args.data_root}...")
    full_dataset = HW4TrainDataset(args.data_root, patch_size=args.patch_size)
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    print(f"Training set: {len(train_dataset)} | Validation set: {len(val_dataset)}")
    
    # 创建数据加载器
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    # 加载或创建模型
    resume_ckpt = None
    if args.auto_resume:
        resume_ckpt = find_latest_checkpoint(ckpt_dir)
        if resume_ckpt:
            print(f"\n[Auto Resume] Loading from: {resume_ckpt}")
        else:
            print(f"\n[Auto Resume] No checkpoint found, starting from scratch")
    elif args.resume:
        if os.path.exists(args.resume):
            resume_ckpt = args.resume
            print(f"\n[Resume] Loading from: {resume_ckpt}")
        else:
            print(f"[Resume] Checkpoint not found: {args.resume}")
    
    if resume_ckpt:
        model = PromptIRModel.load_from_checkpoint(resume_ckpt)
        print("✓ Model loaded successfully!")
    else:
        model = PromptIRModel(learning_rate=args.lr)
        print("✓ Model initialized from scratch")
    
    # 设置logger - 所有logs都在exp_dir/logs下
    logger = TensorBoardLogger(save_dir=logs_dir, name='')
    
    # 设置checkpoint callback - 所有checkpoints都在exp_dir/ckpt下
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:03d}-{val_psnr:.2f}',
        monitor=args.monitor_metric,
        mode='max' if 'psnr' in args.monitor_metric or 'ssim' in args.monitor_metric else 'min',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True
    )
    
    # 设置early stopping (可选)
    callbacks = [checkpoint_callback]
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=args.monitor_metric,
            mode='max' if 'psnr' in args.monitor_metric else 'min',
            patience=args.early_stopping_patience,
            verbose=True,
            check_finite=True
        )
        callbacks.append(early_stopping)
    
    # 设置trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        val_check_interval=1.0,
        log_every_n_steps=50
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print("Starting training...")
    print(f"Validation frequency: Every epoch")
    print(f"Best model metric: {args.monitor_metric}")
    print("=" * 80 + "\n")
    
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=resume_ckpt if resume_ckpt else None
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Experiment directory: {os.path.abspath(exp_dir)}")
    print(f"├── Checkpoints:  {os.path.abspath(ckpt_dir)}")
    print(f"├── Logs:         {os.path.abspath(logs_dir)}")
    print(f"└── Config:       {os.path.abspath(config_path)}")
    print("\nBest models:")
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith('best-'):
            print(f"  ✓ {f}")
    print(f"  ✓ last.ckpt")
    print("=" * 80)


if __name__ == '__main__':
    main()
