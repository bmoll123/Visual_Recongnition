# ./run.sh 2>&1 | tee training_log_RandomErasing=0.3.txt
#!/bin/bash
set -e

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_handcraft_focal \
    --loss focal \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --resume

echo "=========================================="

python train.py \
    --model_name resnext101_handcraft \
    --run_name resnext101_handcraft_ce \
    --loss ce \
    --use_sampler \
    --epochs 100 \
    --batch_size 64 \
    --resume

echo "=========================================="



