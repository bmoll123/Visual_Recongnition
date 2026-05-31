#!/bin/bash

# 一键训练和推理脚本 - Hw4 Image Restoration

set -e

# 配置参数
DATA_ROOT="dataset/hw4_realse_dataset/train"
TEST_ROOT="dataset/hw4_realse_dataset/test"
CKPT_DIR="train_ckpt"
OUTPUT_FILE="pred.npz"

# 默认参数
BATCH_SIZE=8
EPOCHS=150
NUM_GPUS=1
TRAIN_ONLY=false
INFER_ONLY=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --infer-only)
            INFER_ONLY=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Hw4 Image Restoration with PromptIR"
echo "=========================================="

# 检查必要的目录
if [ "$INFER_ONLY" = false ]; then
    if [ ! -d "$DATA_ROOT" ]; then
        echo "Error: Training data not found at $DATA_ROOT"
        exit 1
    fi
    echo "✓ Training data found"
fi

if [ ! -d "$TEST_ROOT" ]; then
    echo "Error: Test data not found at $TEST_ROOT"
    exit 1
fi
echo "✓ Test data found"

mkdir -p "$CKPT_DIR"

# 训练阶段
if [ "$INFER_ONLY" = false ]; then
    echo ""
    echo "========== TRAINING PHASE =========="
    echo "Batch size: $BATCH_SIZE, Epochs: $EPOCHS, GPUs: $NUM_GPUS"
    echo ""
    
    python train_hw4.py \
        --data_root "$DATA_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --num_gpus "$NUM_GPUS" \
        --ckpt_dir "$CKPT_DIR"
    
    echo "✓ Training completed!"
fi

# 推理阶段
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "========== INFERENCE PHASE =========="
    
    # 查找最新的checkpoint
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -n1)
    
    if [ -z "$LATEST_CKPT" ]; then
        echo "Error: No checkpoint found in $CKPT_DIR"
        exit 1
    fi
    
    echo "Using checkpoint: $LATEST_CKPT"
    
    python infer_hw4.py \
        --test_root "$TEST_ROOT" \
        --ckpt_path "$LATEST_CKPT" \
        --output_path "$OUTPUT_FILE"
    
    echo ""
    echo "✓ Inference completed!"
    
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Output: $OUTPUT_FILE ($FILE_SIZE)"
    fi
fi

echo ""
echo "=========================================="
echo "  All done! Ready for submission."
echo "=========================================="
