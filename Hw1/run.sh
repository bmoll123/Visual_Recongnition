# ./run.sh 2>&1 | tee training_log_RandomErasing=0.3.txt
#!/bin/bash
set -e


echo "=========================================="

echo -e "resnext101_handcraft + ce + weighted_sampler"
python train.py \
    --model_name resnext101_handcraft \
    --run_name weighted_sampler_ce \
    --loss ce \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --resume

echo "=========================================="

echo -e "resnext101_handcraft + focal + weighted_sampler"
python train.py \
    --model_name resnext101_handcraft \
    --run_name weighted_sampler_focal \
    --loss focal \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --resume

echo "=========================================="

echo -e "resnext101_handcraft + focal "
python train.py \
    --model_name resnext101_handcraft \
    --run_name focal  \
    --loss focal \
    --epochs 100 \
    --batch_size 64 \
    --resume

echo "=========================================="

echo -e "resnext101_handcraft + ce "
python train.py \
    --model_name resnext101_handcraft \
    --run_name ce \
    --loss ce \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --resume