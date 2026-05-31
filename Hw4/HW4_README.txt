================================================================================
                  Hw4 Image Restoration - Implementation Complete
================================================================================

Dear User,

You now have everything ready to complete your Hw4 Image Restoration assignment!

================================================================================
                              📋 New Files Created
================================================================================

1. utils/hw4_dataset_utils.py
   - Custom data loaders for HW4 dataset (Rain & Snow)
   - HW4TrainDataset: For training with matched degraded/clean pairs
   - HW4TestDataset: For testing on 100 test images

2. train_hw4.py
   - Training script using PyTorch Lightning
   - Supports single/multi-GPU training
   - Auto checkpoint saving and logging

3. infer_hw4.py
   - Inference script to generate pred.npz
   - Automatically converts output to correct format
   - Saves all restored images in required dictionary format

4. run_hw4.sh
   - One-click script to run train+inference
   - Supports various options (--batch-size, --epochs, --gpus, etc.)
   - Automatic checkpoint discovery

5. QUICK_START.md
   - Quick reference for common commands
   - Essential troubleshooting tips
   - Perfect for quick lookups

6. HW4_GUIDE.md
   - Comprehensive guide with detailed explanations
   - Parameter descriptions and examples
   - Expected runtime information
   - Submission checklist

================================================================================
                           🚀 Quick Start Guide
================================================================================

Option 1: One-Click Execution (RECOMMENDED)
============================================
cd /home/yuyun/Desktop/Visual_Recongnition/Hw4
chmod +x run_hw4.sh
./run_hw4.sh

This will automatically:
✓ Train for 150 epochs
✓ Save checkpoints
✓ Run inference
✓ Generate pred.npz

Expected time: 12-25 hours depending on GPU

---

Option 2: Manual Step-by-Step
==============================
Step 1: Train the model
python train_hw4.py --batch_size 8 --epochs 150

Step 2: Run inference
python infer_hw4.py --ckpt_path train_ckpt/epoch=149-train_loss=0.0000.ckpt

---

Option 3: Custom Parameters
===========================
# Reduce memory usage
python train_hw4.py --batch_size 4 --patch_size 96

# Use multiple GPUs
python train_hw4.py --num_gpus 4

# Quick test (10 epochs)
python train_hw4.py --epochs 10

================================================================================
                            📊 Data Format
================================================================================

Training Data:
- Location: dataset/hw4_realse_dataset/train/
- degraded/: 1600 rain + 1600 snow images
- clean/: corresponding clean images
- Total: 3200 training pairs

Test Data:
- Location: dataset/hw4_realse_dataset/test/degraded/
- 100 images: 0.png, 1.png, ..., 99.png
- Labels not provided (your task!)

Output Format:
- File: pred.npz
- Contains dictionary with keys: '0.png', '1.png', ..., '99.png'
- Values: numpy arrays with shape (3, H, W) and dtype uint8
- Pixel values: 0-255

================================================================================
                         ✅ Pre-Submission Checklist
================================================================================

Before submitting, verify:
□ Training completed (150 epochs)
□ Checkpoints saved in train_ckpt/
□ pred.npz file generated
□ Contains 100 images (0.png to 99.png)
□ Each image has shape (3, H, W)
□ All values are uint8 (0-255)
□ File size is reasonable (10-50 MB)
□ Submitted before deadline

Verification command:
python << 'EOF'
import numpy as np
data = np.load('pred.npz')
print(f"✓ Images: {len(data.files)}")
for key in list(data.files)[:3]:
    img = data[key]
    print(f"  {key}: {img.shape} {img.dtype} [{img.min()}-{img.max()}]")
EOF

================================================================================
                            📝 File Guide
================================================================================

For Quick Questions:
→ Read QUICK_START.md (2 minutes)

For Detailed Information:
→ Read HW4_GUIDE.md (10 minutes)

For Understanding the Code:
→ Look at utils/hw4_dataset_utils.py
→ Look at train_hw4.py
→ Look at infer_hw4.py

For Training Progress:
→ Watch logs/ directory with TensorBoard
→ tensorboard --logdir logs/

================================================================================
                          🔧 Important Notes
================================================================================

1. Data Loading
   - HW4TrainDataset automatically matches degraded/clean pairs
   - File naming: degraded=rain-1.png, clean=rain_clean-1.png
   - Supports both rain and snow in a single model

2. Training
   - Uses PromptIR model with L1 loss
   - AdamW optimizer with learning rate scheduling
   - Warm-up for 15 epochs, then cosine annealing
   - Automatic checkpoint saving every epoch

3. Inference
   - Loads checkpoint and runs on test images
   - Converts float[0,1] to uint8[0,255]
   - Saves in npz format with compression

4. Common Issues
   - OOM? Use: python train_hw4.py --batch_size 4
   - Slow? Check: nvidia-smi (GPU usage)
   - Lost checkpoint? Check: ls -la train_ckpt/

================================================================================
                         💡 Pro Tips
================================================================================

• Run in background: nohup python train_hw4.py > training.log 2>&1 &
• Monitor progress: tail -f training.log
• Check GPU: nvidia-smi
• View checkpoints: ls -lh train_ckpt/
• Test quickly: python train_hw4.py --epochs 10 --batch_size 4

================================================================================
                        🎓 About PromptIR
================================================================================

PromptIR is a Transformer-based image restoration model that:
✓ Supports multiple restoration tasks
✓ Uses prompt-based learning
✓ Achieves state-of-the-art performance
✓ Trains from scratch (no pretraining required)

Your Implementation:
✓ Unified model for Rain and Snow removal
✓ Custom data loaders for your dataset
✓ Proper training pipeline with PyTorch Lightning
✓ Automatic inference and npz generation

================================================================================
                         📞 Need Help?
================================================================================

1. Check the documentation
   - QUICK_START.md for commands
   - HW4_GUIDE.md for detailed info

2. Verify your setup
   - Check data: ls dataset/hw4_realse_dataset/
   - Check conda: conda activate base
   - Check GPU: nvidia-smi

3. Common fixes
   - Restart Python kernel
   - Clear GPU cache: echo "import torch; torch.cuda.empty_cache()" | python
   - Check file permissions: chmod +x run_hw4.sh

================================================================================

Good luck with your assignment! 🚀

You're all set. Simply run:
    cd /home/yuyun/Desktop/Visual_Recongnition/Hw4
    ./run_hw4.sh

Or follow the QUICK_START.md for step-by-step instructions.

================================================================================
