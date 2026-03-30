import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(train_dir):
    class_names = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    num_classes = len(class_names)

    class_counts = []
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        num_files = len(
            [
                f
                for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ]
        )
        class_counts.append(num_files)

    class_counts = np.array(class_counts)
    total_samples = np.sum(class_counts)

    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset):
    targets = []

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        for _, label, _ in dataset:
            targets.append(label)

    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights.astype(np.float32)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=1 * len(sample_weights), replacement=True
    )

    return sampler


def save_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(40, 40))

    annot_labels = np.empty_like(cm, dtype=str)
    annot_labels[:] = ""
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                annot_labels[i, j] = str(cm[i, j])

    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"shrink": 0.8},
    )

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.xlabel("Predicted Labels", fontsize=20, labelpad=20)
    plt.ylabel("True Labels", fontsize=20, labelpad=20)
    plt.title("Confusion Matrix", fontsize=28, pad=20)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_predictions(model, device, test_loader, output_csv_path, run_name):
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, _, img_names in tqdm(test_loader, desc="Generating Predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for img_name, pred in zip(img_names, preds):
                results.append({"image_name": img_name, "pred_label": pred.item()})

    df = pd.DataFrame(results)
    df = df.sort_values(by="image_name").reset_index(drop=True)

    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to: {output_csv_path}")

    output_zip_path = Path(output_csv_path).parent / f"{run_name}.zip"
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv_path, arcname=Path(output_csv_path).name)
    print(f"Zipped prediction saved to: {output_zip_path}\n")
