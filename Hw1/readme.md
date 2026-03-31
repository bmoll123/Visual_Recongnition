# NYCU Computer Vision 2026 HW1

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton

* Task Objective: Perform 100-class image classification on natural RGB images.
* Dataset Size: The training set contains 21,024 images, and the test set contains 2,344 images.
* Core Architecture: Built upon the ResNet foundation, I proposed an improved model based on the ResNeXt-101 architecture. Specifically, I integrated the Squeeze-and-Excitation (SE) channel attention mechanism to help the network focus on critical informative features, and incorporated Dropout to prevent overfitting and enhance overall model robustness.
* Handling Class Imbalance: Observing an uneven class distribution within the training data, a Weighted Sampler is utilized during the data loading phase. This ensures a more balanced class representation in each training batch, enabling more stable model convergence.


## Environment Setup

```
conda create -n python3.9 python=3.9 -y

conda activate python3.9

pip install -r requirements.txt
```

## Usage

### Training + Inference
```
python train.py \
    --model_name resnext101_se \
    --loss ce \
    --scheduler cosine \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume
```

### Only Inference
```
python train.py \
    --test \
    --model_name resnext101_se \
    --load_weight ./Results/Run_resnext101_se_ce_sampler/weights/last.pt
```

## Performance Snapshot
<img width="1155" height="46" alt="截圖 2026-03-31 晚上9 08 03" src="https://github.com/user-attachments/assets/c094d377-5efd-4f79-884c-4a7b789b53ec" />

