# NYCU Computer Vision 2026 HW3

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton




## Environment Setup

```
conda create -n promptir python=3.9 -y
conda activate promptir

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4 einops==0.6.0 timm==0.6.12 wandb==0.13.9 scipy scikit-image scikit-learn matplotlib pandas opencv-python yapf

pip install -U openmim
mim install mmcv-full==1.7.1

pip install warmup_scheduler
pip install "numpy<2"
pip install lightning
pip install accelerate deepspeed
pip install -U 'tensorboard'
pip install -U 'tensorboardX'
pip install scikit-video
```

## Usage

### Training
```
 python train_hw4.py --epochs 150 --batch_size 4 --exp_dir ./results/0531_origin --auto-resume
 ```
--resume

### Inference
```

```

## Performance Snapshot
