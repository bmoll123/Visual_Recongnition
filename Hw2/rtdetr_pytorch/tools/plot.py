import json
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR 訓練曲線繪製工具 (精簡版)")
    # 設定參數
    parser.add_argument(
        "--log",
        type=str,
        default="/home/cvml-3/yy/114_2/Visual_Recongition/rtdetr/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco/log.txt",
        help="輸入的 log 檔案路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ciou_se_rtdter_training_curves.png",
        help="輸出的圖片檔名",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RT-DETR Training Analysis",
        help="圖表的大標題",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=50,
        help="最大繪製 Epoch 數 (預設: 50)",
    )
    return parser.parse_args()


def load_logs(log_path):
    if not os.path.exists(log_path):
        print(f"❌ 錯誤：找不到檔案 {log_path}")
        return []

    data_dict = {}
    with open(log_path, "r") as f:
        for line in f:
            try:
                log = json.loads(line)
                epoch = log["epoch"]
                data_dict[epoch] = log
            except (json.JSONDecodeError, KeyError):
                continue

    sorted_epochs = sorted(data_dict.keys())
    return [data_dict[e] for e in sorted_epochs]


def plot_curves(logs, output_name, main_title):
    if not logs:
        print("⚠️ 沒有可用的數據進行繪圖。")
        return

    epochs = [log["epoch"] for log in logs]
    total_loss = [log["train_loss"] for log in logs]
    # 取得 mAP 0.5:0.95 (通常是列表中的第一個元素)
    mAP_05_95 = [log["test_coco_eval_bbox"][0] for log in logs]

    # 開始繪圖
    plt.figure(figsize=(14, 6))
    plt.suptitle(main_title, fontsize=16)

    # 子圖 1: Total Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, total_loss, label="Total Loss", color="#1f77b4", linewidth=2)
    plt.title("Total Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.xlim(0, max(epochs))
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # 子圖 2: Validation mAP (0.5:0.95)
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        mAP_05_95,
        label="mAP @ 0.5:0.95",
        color="#ff7f0e",
        marker="s",
        markersize=4,
    )
    plt.title("Validation Precision (mAP 0.5:0.95)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.xlim(0, max(epochs))
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name)
    print(f"✅ 成功！已儲存精簡版曲線圖至：{output_name}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    raw_logs = load_logs(args.log)

    # 過濾邏輯：只保留指定範圍內的 logs
    filtered_logs = [log for log in raw_logs if log["epoch"] <= args.max_epoch]

    plot_curves(filtered_logs, args.output, args.title)
