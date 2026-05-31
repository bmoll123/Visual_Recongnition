為您寫好了一份結構完整、排版清晰的 `README.md`。這份說明書針對您提供的高級實驗管理指令腳本（`train_hw4.py`）進行了詳細的參數說明與範例演示，方便您自己、助教或組員快速上手。

---

# PromptIR 實驗管理與訓練指南 (HW4)

本腳本是專為 **HW4 影像修復任務** 打造的完整實驗管理訓練系統。每次執行都會將所有輸出（包括模型權重、訓練日誌、實驗配置）結構化地儲存至指定的實驗目錄中，確保實驗過程 100% 可追溯。

## 📌 實驗目錄結構

當您啟動一個實驗時（例如指定 `--exp_dir experiments/exp_001`），腳本會自動為您建立以下結構：

```text
experiments/exp_001/
├── config.yaml          # 本次實驗的完整參數與執行時間記錄
├── ckpt/                # 模型權重資料夾
│   ├── best-epoch=012-val_psnr=28.50.ckpt  # 性能前 K 佳的模型
│   └── last.ckpt        # 最新一輪的權重（用於中斷恢復）
└── logs/                # TensorBoard 訓練日誌資料夾
    └── version_0/
        └── events.out.tfevents...

```

---

## 🚀 常用指令範例

### 1. 第一次啟動訓練（全新實驗）

指定實驗目錄為 `exp_001`，設定 Batch Size 為 4 跑 150 個 Epoch：

```bash
python train_hw4.py --exp_dir experiments/exp_001 --epochs 150 --batch_size 4

```

### 2. 中斷後自動恢復訓練（Auto Resume）

如果訓練到一半斷網、斷電或被系統砍掉，只要加上 `--auto_resume`，腳本會自動去 `ckpt/` 資料夾尋找最新的進度並無縫接軌繼續訓練：

```bash
python train_hw4.py --exp_dir experiments/exp_001 --auto_resume --epochs 150

```

### 3. 從特定 Checkpoint 載入權重微調

如果您想指定特定的權重檔案進行載入（例如改跑不同的實驗，但想用之前的預訓練權重）：

```bash
python train_hw4.py --exp_dir experiments/exp_002_finetune --resume experiments/exp_001/ckpt/best-epoch=050-val_psnr=29.20.ckpt --epochs 50

```

### 4. 多顯示卡平行訓練（Multi-GPU DDP）

如果您實驗室的伺服器有多張顯卡（例如 2 張 GPU），腳本已內建 PyTorch Lightning 的 DDP 分散式架構，直接指定數量即可倍速訓練：

```bash
python train_hw4.py --exp_dir experiments/exp_multi_gpu --num_gpus 2 --batch_size 8

```

---

## 🎛️ 完整參數說明表

您可以透過 `python train_hw4.py --help` 查看所有參數，以下為重點參數彙整：

### 1. 實驗設置 (Experiment Settings)

| 參數 | 型態 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--exp_dir` | `str` | **(必填)** | 實驗儲存的根目錄路徑（例如 `exps/baseline`） |
| `--exp_name` | `str` | `promptir_hw4` | 用於 TensorBoard 紀錄的實驗名稱識別 |

### 2. 資料集設置 (Data Settings)

| 參數 | 型態 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--data_root` | `str` | `dataset/hw4_realse_dataset/train` | 訓練資料集的根目錄路徑 |
| `--patch_size` | `int` | `128` | 訓練時隨機裁切的影像 Patch 大小 |
| `--batch_size` | `int` | `8` | 每張 GPU 的單次訓練批次量 |
| `--num_workers` | `int` | `4` | DataLoader 的多線程讀取數量 |
| `--val_split` | `float` | `0.1` | 切分驗證集的比例（`0.1` 代表 10% 資料用作驗證） |

### 3. 訓練與優化器設置 (Training Settings)

| 參數 | 型態 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--epochs` | `int` | `150` | 總訓練輪數 |
| `--lr` | `float` | `2e-4` | 初始學習率（會搭配 15 Epochs 的 Warmup 與 Cosine 退火） |
| `--num_gpus` | `int` | `1` | 使用的 GPU 數量 |

### 4. 權重保存與早停機制 (Checkpoint & Early Stopping)

| 參數 | 型態 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--monitor_metric` | `str` | `val_psnr` | 監控的核心指標，可選：`val_psnr`, `val_loss`, `val_ssim` |
| `--save_top_k` | `int` | `3` | 自動保留效能最好的前 K 個模型檔案 |
| `--early_stopping_patience` | `int` | `20` | 超過幾輪指標沒有改善就提早結束訓練（`0` 代表關閉此功能） |

---

## 📊 訓練日誌視覺化 (TensorBoard)

訓練開啟後，您可以隨時開另一個終端機視窗，啟動 TensorBoard 來即時監控 `Loss`、`PSNR` 與 `SSIM` 的曲線變化：

```bash
tensorboard --logdir experiments/exp_001/logs/

```

接著打開瀏覽器輸入 `http://localhost:6006` 即可看到精美的圖表。