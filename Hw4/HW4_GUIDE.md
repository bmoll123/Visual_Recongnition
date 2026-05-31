# Hw4 Image Restoration with PromptIR - 完整指南

## 🎯 项目概述

本项目使用 **PromptIR** 模型完成图像恢复任务，支持同时处理 **Rain** 和 **Snow** 两种类型的图像退化。

---

## 📂 数据集结构

```
dataset/hw4_realse_dataset/
├── train/
│   ├── degraded/     # 退化图像 (3200张)
│   │   ├── rain-1.png ~ rain-1600.png
│   │   └── snow-1.png ~ snow-1600.png
│   └── clean/        # 干净图像 (3200张)
│       ├── rain_clean-1.png ~ rain_clean-1600.png
│       └── snow_clean-1.png ~ snow_clean-1600.png
└── test/
    └── degraded/     # 测试图像 (100张)
        └── 0.png ~ 99.png
```

---

## 🛠️ 核心文件说明

| 文件 | 功能 |
|------|------|
| `utils/hw4_dataset_utils.py` | 数据加载器（新建） |
| `train_hw4.py` | 训练脚本（新建） |
| `infer_hw4.py` | 推理脚本（新建） |
| `run_hw4.sh` | 一键运行脚本（新建） |
| `net/model.py` | PromptIR模型 |
| `utils/image_utils.py` | 图像处理工具 |

---

## 🚀 快速开始

### 方式一：一键运行（推荐）
```bash
chmod +x run_hw4.sh
./run_hw4.sh
```

### 方式二：分步运行

**仅训练**
```bash
python train_hw4.py --epochs 150 --batch_size 8
```

**仅推理**
```bash
python infer_hw4.py --ckpt_path train_ckpt/epoch=149-train_loss=0.0000.ckpt
```

---

## 📋 参数说明

### train_hw4.py 主要参数

```bash
python train_hw4.py \
    --data_root <路径>           # 训练数据根目录（默认：dataset/hw4_realse_dataset/train）
    --patch_size <数字>          # Patch大小（默认：128）
    --batch_size <数字>          # 批大小（默认：8）
    --num_workers <数字>         # 数据加载线程（默认：4）
    --epochs <数字>              # 训练轮数（默认：150）
    --lr <浮点数>                # 学习率（默认：2e-4）
    --num_gpus <数字>            # GPU数量（默认：1）
    --ckpt_dir <路径>            # 检查点保存目录（默认：train_ckpt）
    --use_wandb                  # 使用Weights & Biases日志（可选）
```

### infer_hw4.py 主要参数

```bash
python infer_hw4.py \
    --test_root <路径>           # 测试数据根目录（默认：dataset/hw4_realse_dataset/test）
    --ckpt_path <路径>           # 模型检查点路径
    --output_path <路径>         # 输出npz文件路径（默认：pred.npz）
    --batch_size <数字>          # 推理批大小（默认：1）
    --num_workers <数字>         # 数据加载线程（默认：0）
    --cuda <数字>                # GPU设备ID（默认：0）
```

---

## 💻 常用命令

### 内存受限情况
```bash
# 减小batch_size
python train_hw4.py --batch_size 4

# 或减小patch_size
python train_hw4.py --patch_size 96
```

### 加速训练
```bash
# 使用多GPU
python train_hw4.py --num_gpus 4

# 或增加batch_size
python train_hw4.py --batch_size 16
```

### 测试运行
```bash
# 快速验证（只训练10个epoch）
python train_hw4.py --epochs 10
```

### 监控训练
```bash
# TensorBoard
tensorboard --logdir logs/

# 访问 http://localhost:6006
```

---

## 📊 数据加载器详解

### HW4TrainDataset
- 自动匹配degraded和clean图像
- 支持随机patch裁剪
- 返回格式：`([name, de_id], degraded_tensor, clean_tensor)`
  - `name`: 图像名称（如'rain-1'）
  - `de_id`: 退化类型ID（0=rain, 1=snow）

### HW4TestDataset
- 按顺序加载测试图像
- 返回格式：`([image_name], degraded_tensor)`
  - `image_name`: 图像文件名（如'0.png'）

---

## 📤 输出格式

### pred.npz 文件结构
```python
{
    '0.png': numpy.ndarray(shape=(3, H, W), dtype=uint8),
    '1.png': numpy.ndarray(shape=(3, H, W), dtype=uint8),
    ...
    '99.png': numpy.ndarray(shape=(3, H, W), dtype=uint8)
}
```

**验证方法**：
```python
import numpy as np
data = np.load('pred.npz')
print(f"Number of images: {len(data.files)}")  # 应该是100
print(f"Image shape: {data['0.png'].shape}")    # 应该是 (3, H, W)
print(f"Data type: {data['0.png'].dtype}")      # 应该是 uint8
print(f"Value range: {data['0.png'].min()}-{data['0.png'].max()}")  # 应该是 0-255
```

---

## 🐛 常见问题

### Q1: 显存不足 (Out of Memory)
```bash
python train_hw4.py --batch_size 4 --patch_size 96
```

### Q2: 找不到checkpoint
```bash
# 检查checkpoint目录
ls -la train_ckpt/

# 确保训练已完成
# 使用最新的checkpoint
ls -t train_ckpt/*.ckpt | head -1
```

### Q3: 推理很慢
```bash
# 使用更大的batch_size
python infer_hw4.py --batch_size 8

# 检查GPU
nvidia-smi
```

### Q4: npz文件损坏
```bash
# 重新运行推理
rm pred.npz
python infer_hw4.py --ckpt_path <正确路径>
```

---

## ⏱️ 预期耗时

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 环境设置 | 1-2 min | 一次性 |
| 单个epoch | 5-10 min | 取决于GPU |
| 150个epoch | 12-25 hr | RTX3090约12hr |
| 推理100张 | 2-5 min | 批处理 |

---

## ✅ 提交检查清单

- [ ] 成功运行150个epoch的训练
- [ ] 在train_ckpt/目录中有checkpoint文件
- [ ] 生成了pred.npz文件
- [ ] pred.npz包含100个图像（0.png-99.png）
- [ ] 每个图像shape为(3, H, W)，dtype为uint8
- [ ] 像素值范围在[0, 255]之间
- [ ] 文件大小合理（通常10-50MB）
- [ ] 提交时间在截止日期前

---

## 🔧 高级选项

### 自定义训练参数
```bash
python train_hw4.py \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 12 \
    --patch_size 256
```

### 使用W&B日志
```bash
wandb login
python train_hw4.py --use_wandb --wandb_project my-project
```

### 后台运行
```bash
# 保存日志
nohup python train_hw4.py > training.log 2>&1 &

# 查看日志
tail -f training.log
```

---

## 📝 文件清单

运行完成后会生成以下文件：
```
train_ckpt/
├── epoch=000-train_loss=0.xxxx.ckpt
├── epoch=001-train_loss=0.xxxx.ckpt
├── ...
└── epoch=149-train_loss=0.xxxx.ckpt    # 最新checkpoint

logs/
└── （TensorBoard日志）

pred.npz                                 # 最终提交文件
```

---

## 💡 优化建议

1. **数据增强**：可在HW4TrainDataset中添加随机翻转、旋转
2. **学习率调整**：根据loss曲线调整初始学习率
3. **Patch大小**：较大的patch(256)可能提升性能但需更多显存
4. **Batch size**：增加batch_size可加速训练但需更多显存

---

## 🎓 PromptIR模型说明

**关键特性**：
- Transformer架构的图像恢复模型
- Prompt-based图像复原
- 支持多种图像恢复任务

**本项目修改**：
- 适配Rain和Snow两种退化类型
- 统一在单个模型中训练

---

## 📞 获取帮助

1. 查看完整日志：`tail -f training.log`
2. 检查数据路径：`ls dataset/hw4_realse_dataset/train/degraded/ | head`
3. 验证GPU：`nvidia-smi`
4. 查看模型参数：`python -c "from net.model import PromptIR; m=PromptIR(); print(sum(p.numel() for p in m.parameters()))"`

---

**更新时间**: 2026-06-01
**版本**: 1.0

Good luck! 🚀
