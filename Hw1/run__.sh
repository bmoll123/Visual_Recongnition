#!/bin/bash
set -e

echo "=========================================="

python train.py \
    --model_name resnet50 \
    --run_name resnet50 \
    --loss ce \
    --epochs 100 \
    --batch_size 64 \
    --resume

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101 \
    --loss ce \
    --epochs 100 \
    --batch_size 64 \
    --resume

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_weightsampler \
    --use_sampler \
    --loss ce \
    --epochs 100 \
    --batch_size 64 \
    --resume

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_weightsampler \
    --use_sampler \
    --loss focal \
    --epochs 100 \
    --batch_size 64 \
    --resume

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_weightsampler_150 \
    --use_sampler \
    --loss ce \
    --epochs 150 \
    --batch_size 64 \
    --resume

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_weightsampler_150 \
    --use_sampler \
    --loss focal \
    --epochs 150 \
    --batch_size 64 \
    --resume

