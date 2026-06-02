# NYCU Computer Vision 2026 HW3

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton

### Task Objective
This assignment focuses on **Image Restoration**. The proposed model takes degraded images corrupted by either rain or snow as input and predicts the corresponding clean, high-quality images. The primary challenge is to effectively handle multiple, unknown degradation types within a single, unified framework.

### Dataset Description
The dataset consists of paired degraded and ground-truth clean images divided into the following splits:

| Dataset Split | Rain Images | Snow Images | Total Images |
| :--- | :---: | :---: | :---: |
| **Training Set** | 1,440 | 1,440 | **2,880** |
| **Validation Set** | 160 | 160 | **320** |
| **Testing Set** | 50 | 50 | **100** |

### Core Architecture
The foundational framework of this project is **PromptIR** (Prompting for All-in-One Image Restoration). 

* **Prompt-based Learning:** PromptIR utilizes a prompt-based learning formulation to dynamically adapt to different image degradations without relying on prior knowledge of the degradation type (blind restoration).
* **Advanced Customization:** To further improve the structural and high-frequency restoration performance, the default optimization objective was enhanced by introducing **Edge-preservation** and **Frequency-domain constraints** into a composite loss function, significantly boosting the final PSNR performance.


## Environment Setup

```
conda create -n promptir python=3.9 -y
conda activate promptir

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4 einops==0.6.0 timm==0.6.12 wandb==0.13.9 scipy scikit-image scikit-learn matplotlib pandas opencv-python yapf

pip install -U openmim
mim install mmcv-full==1.7.1

pip install warmup_scheduler
pip install "numpy<2"
pip install lightning
pip install accelerate deepspeed
pip install -U 'tensorboard'
pip install -U 'tensorboardX'
pip install scikit-video
```

## Usage

### Training
```
python train_hw4.py --epochs 150 --batch_size 4 --exp_dir ./results/ --auto-resume
```

### Inference
```
python infer_hw4.py --ckpt_path ./results/last.ckpt --output_path ./results/inference/pred.npz
```

## Performance Snapshot
<img width="1070" height="41" alt="截圖 2026-06-02 晚上8 59 31" src="https://github.com/user-attachments/assets/2eb2755c-3631-4bb3-b3b8-3239b50f5b09" />
