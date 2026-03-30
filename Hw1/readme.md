# NYCU Computer Vision 2026 HW1

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton

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
    --model_name resnext101_handcraft \
    --loss focal \
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
    --model_name resnext101_handcraft \
    --load_weight ./Results/Run_resnext101_handcraft_focal_sampler/weights/last.pt
```

## Performance Snapshot