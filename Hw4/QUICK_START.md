# Hw4 快速命令指南

## 📋 前置准备

### 1. 激活环境
```bash
conda activate base  # 或你的环境名称
```

### 2. 进入项目目录
```bash
cd /home/yuyun/Desktop/Visual_Recongnition/Hw4
```

---

## 🚀 最简单的方式：一键运行

```bash
# 给脚本添加执行权限
chmod +x run_hw4.sh

# 运行脚本（自动训练+推理）
./run_hw4.sh
```

这会自动：
1. ✅ 训练150个epoch
2. ✅ 使用最新的checkpoint进行推理
3. ✅ 生成 `pred.npz` 文件

---

## 🔧 分步骤运行

### 方案 A：默认参数（推荐）

**只训练**
```bash
python train_hw4.py
```

**只推理**（训练完成后）
```bash
python infer_hw4.py
```

---

### 方案 B：自定义参数

**训练 - 自定义参数**
```bash
python train_hw4.py \
    --data_root dataset/hw4_realse_dataset/train \
    --batch_size 8 \
    --patch_size 128 \
    --epochs 150 \
    --lr 2e-4 \
    --num_gpus 1 \
    --ckpt_dir train_ckpt
```

**推理 - 自定义checkpoint**
```bash
python infer_hw4.py \
    --test_root dataset/hw4_realse_dataset/test \
    --ckpt_path train_ckpt/epoch=149-train_loss=0.0000.ckpt \
    --output_path pred.npz
```

---

## 📊 常用命令组合

### 减少显存占用（RTX2080或更低）
```bash
python train_hw4.py --batch_size 4
```

### 使用多GPU加速
```bash
python train_hw4.py --num_gpus 4
```

### 快速测试（少轮数）
```bash
python train_hw4.py --epochs 10
```

---

## 🔍 验证和调试

### 检查生成的pred.npz
```bash
python << 'EOF'
import numpy as np
data = np.load('pred.npz')
print(f"✓ Image count: {len(data.files)}")
for key in list(data.files)[:3]:
    img = data[key]
    print(f"  {key}: {img.shape} {img.dtype}")
EOF
```

### 查看训练checkpoint
```bash
ls -lh train_ckpt/
```

### 监控GPU使用
```bash
nvidia-smi
```

---

## ✅ 提交前检查清单

- [ ] 训练完成（epoch=150）
- [ ] 生成了 `pred.npz` 文件
- [ ] 运行验证代码显示100个图像
- [ ] npz文件大小 > 5MB

---

## 💡 提示

- 第一次运行可能较慢（模型编译、数据加载等）
- 每个epoch大约耗时 **5-10分钟**（取决于GPU）
- 150个epoch总耗时约 **12-25小时**
- 推荐在后台运行：`nohup python train_hw4.py > training.log 2>&1 &`

---

Good luck! 🚀
