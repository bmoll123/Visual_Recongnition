# NYCU Computer Vision 2026 HW3

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton

* Task Objective: This assignment performs instance segmentation on colored medical images. The model takes RGB medical images as input and predicts both the correct class and the precise segmentation masks for four types of cells (class1, class2, class3, class4). The primary evaluation metric for this task is AP50.
* Dataset Size: 
    * Training / Validation Set: 209 images
    * Testing Set: 101 images

* Core Architecture: Built upon the Mask R-CNN foundation, I proposed an improved model based on the Hybrid Task Cascade (HTC) for Instance Segmentation:
    * Neck Structure Enhancement (FPN -> PAFPN): I replaced the default Feature Pyramid Network (FPN) with a Path Aggregation Network (PAFPN). While FPN only has a top-down pathway, PAFPN introduces an additional bottom-up path. This modification shortens the information flow, allowing low-level spatial features (such as cell edges and fine contours) to be directly transmitted to the higher-level feature maps.



## Environment Setup

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install -v -e .
```

## Usage

### Training
```
OPENCV_LOG_LEVEL=OFF python tools/train.py \
    tools_my/htc_config.py \
    --work-dir results \
    --resume
```

### Inference
```
OPENCV_LOG_LEVEL=OFF python tools/test.py \
    tools_my/htc_config.py \
    results/epoch_50.pth \
    --work-dir results \
    --do-val
```

## Performance Snapshot
<img width="1129" height="40" alt="截圖 2026-05-12 晚上10 36 14" src="https://github.com/user-attachments/assets/86ef6e53-5d78-4386-8695-6e30dc609558" />

