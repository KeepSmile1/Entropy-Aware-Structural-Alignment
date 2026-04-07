#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Curve Plotting Script for Open-Source Project
Generates:
    - Epoch vs Accuracy curve
    - Log-scaled training loss curve
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict


def plot_accuracy_curve(
    txt_path: str = "record.txt",
    save_path: str = "Epoch_vs_Accuracy.png",
    dpi: int = 300
) -> None:
    """Plot accuracy curve from training record."""
    epochs, accuracies = [], []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Epoch" in line and "ACC" in line:
                parts = line.strip().split("|")
                epoch = int(parts[0].split(":")[1].strip())
                acc = float(parts[1].split(":")[1].strip())
                epochs.append(epoch)
                accuracies.append(acc)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linewidth=2, label="Accuracy")
    plt.title("Epoch vs Accuracy", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def compute_average_loss(
    log_path: str = "train_log.txt",
    save_path: str = "average_losses.txt",
    interval: int = 4
) -> None:
    """Compute average loss every N steps and save to txt."""
    with open(log_path, "r", encoding="utf-8") as f:
        log_data = f.read()

    pattern = r"epoch: (\d+) .*? loss_rec: ([\d\.]+)"
    matches = re.findall(pattern, log_data)
    loss_buffer = []

    with open(save_path, "w", encoding="utf-8") as f_out:
        for idx, (ep, loss) in enumerate(matches):
            loss_buffer.append(float(loss))
            if (idx + 1) % interval == 0:
                avg_loss = sum(loss_buffer) / len(loss_buffer)
                f_out.write(f"{avg_loss:.6f}\n")
                loss_buffer = []


def plot_log_loss_curve(
    loss_path: str = "average_losses.txt",
    save_path: str = "loss_curve_log_scaled.png",
    dpi: int = 300
) -> None:
    """Plot log-scaled loss curve with epoch markers."""
    loss_values = []
    with open(loss_path, "r", encoding="utf-8") as f:
        for line in f:
            loss_values.append(float(line.strip()))

    x = list(range(1, len(loss_values) + 1))
    log_loss = [np.log2(v if v > 0 else 1e-10) for v in loss_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x, log_loss, marker='o', label="Log-scaled Loss")

    xticks_pos = list(range(0, len(x), 4))
    xticks_lab = [f"ep {i // 4}" for i in xticks_pos]
    plt.xticks(ticks=[i + 1 for i in xticks_pos], labels=xticks_lab)

    plt.title("Log-Scaled Loss Curve", fontsize=16)
    plt.xlabel("Step (Epoch)", fontsize=14)
    plt.ylabel("Log Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    # Run all pipeline
    plot_accuracy_curve = './history/exp_name/record.txt'  # 替换为实际路径
    plot_accuracy_curve(plot_accuracy_curve, save_path="./fig/Epoch_vs_Accuracy.png")

    log_path = './history/exp_name/train_log.txt'
    compute_average_loss(log_path=log_path, save_path="./fig/average_losses.txt")

    loss_path = "./fig/average_losses.txt"
    plot_log_loss_curve(loss_path=loss_path, save_path="./fig/loss_curve_log_scaled.png")