import json
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 設定區塊
# ==========================================
log_file_path = (
    "../results/260512_50e_fafpn/20260512_213941/vis_data/20260512_213941.json"
)
output_image_name = "../results/260512_50e_fafpn/training_loss_curve_epoch.png"

# ==========================================
# 2. 讀取與解析檔案
# ==========================================
parsed_logs = []

if not os.path.exists(log_file_path):
    print(f"錯誤：找不到檔案 {log_file_path}")
else:
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                log_item = json.loads(line.strip())
                if "loss" in log_item:
                    parsed_logs.append(log_item)
            except json.JSONDecodeError:
                continue

if not parsed_logs:
    print("未抓取到任何有效的訓練數據。")
else:
    # ==========================================
    # 3. 提取數據 (將 step 改為 epoch)
    # ==========================================
    # 改為抓取 epoch 鍵值
    epochs = [log["epoch"] for log in parsed_logs]
    total_loss = [log["loss"] for log in parsed_logs]

    s0_mask = [log.get("s0.loss_mask", 0) for log in parsed_logs]
    s1_mask = [log.get("s1.loss_mask", 0) for log in parsed_logs]
    s2_mask = [log.get("s2.loss_mask", 0) for log in parsed_logs]

    # ==========================================
    # 4. 開始繪圖
    # ==========================================
    plt.figure(figsize=(12, 7))

    # 繪製總損失
    plt.plot(
        epochs,  # 橫軸改為 epoch
        total_loss,
        label="Total Loss",
        color="black",
        linewidth=1.5,
        linestyle="--",
    )

    # 繪製各階段 Mask Loss
    plt.plot(epochs, s0_mask, label="Stage 0 Mask Loss", alpha=0.8)
    plt.plot(epochs, s1_mask, label="Stage 1 Mask Loss", alpha=0.8)
    plt.plot(epochs, s2_mask, label="Stage 2 Mask Loss", alpha=0.8)

    # 圖表裝飾
    plt.title(
        f"Training Loss Curve (by Epoch)\nSource: {os.path.basename(log_file_path)}",
        fontsize=14,
    )
    plt.xlabel("Epoch", fontsize=12)  # 標籤改為 Epoch
    plt.ylabel("Loss Value", fontsize=12)

    # 讓 X 軸的刻度更漂亮 (視情況顯示)
    # plt.xticks(range(min(epochs), max(epochs) + 1, 5)) # 每 5 個 epoch 顯示一個刻度

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # 5. 儲存與顯示
    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"圖表已成功儲存為：{output_image_name}")
    plt.show()
