"""
train_unet.py

Trains our small U-Net to restore degraded images back toward their
clean originals. Uses images NOT in the test set (coins, gravel are
held out) for training.

After training, evaluates on the held-out test images and saves:
- the trained model weights (models/unet_restoration.pth)
- restored test images (outputs/restored/unet/)
- a training loss curve (outputs/plots/unet_training_loss.png)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from dataset import RestorationDataset
from unet import SmallUNet
from metrics import evaluate_all

MODELS_DIR = "models"
RESTORED_DIR = os.path.join("outputs", "restored", "unet")
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 0.001


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    train_dataset = RestorationDataset(split="train")
    test_dataset = RestorationDataset(split="test")

    print("Training pairs:", len(train_dataset))
    print("Test pairs:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SmallUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for degraded, clean, filenames in train_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(degraded)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print("Epoch " + str(epoch) + "/" + str(EPOCHS) + " - Loss: " + str(round(avg_loss, 5)))

    model_path = os.path.join(MODELS_DIR, "unet_restoration.pth")
    torch.save(model.state_dict(), model_path)
    print("\nModel saved to:", model_path)

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (L1)")
    plt.title("U-Net Training Loss")
    plot_path = os.path.join(PLOTS_DIR, "unet_training_loss.png")
    plt.savefig(plot_path)
    print("Loss plot saved to:", plot_path)

    print("\n--- Evaluating on held-out test images ---\n")
    model.eval()
    all_results = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            degraded, clean, filename = test_dataset[i]
            degraded_batch = degraded.unsqueeze(0).to(device)
            output = model(degraded_batch)

            restored_arr = output.squeeze().cpu().numpy()
            restored_arr = np.clip(restored_arr * 255.0, 0, 255).astype(np.uint8)

            clean_arr = clean.squeeze().numpy()
            clean_arr = np.clip(clean_arr * 255.0, 0, 255).astype(np.uint8)

            out_path = os.path.join(RESTORED_DIR, filename)
            Image.fromarray(restored_arr).save(out_path)

            metrics = evaluate_all(clean_arr, restored_arr)
            metrics["filename"] = filename
            all_results.append(metrics)

            print(filename + ": PSNR=" + str(round(metrics["PSNR"], 2)) + " dB, SSIM=" + str(round(metrics["SSIM"], 4)) + ", MAE=" + str(round(metrics["MAE"], 2)))

    if all_results:
        psnr_values = [r["PSNR"] for r in all_results]
        ssim_values = [r["SSIM"] for r in all_results]
        mae_values = [r["MAE"] for r in all_results]

        print("\n--- AVERAGE (U-Net Restoration, test set) ---")
        print("PSNR: " + str(round(np.mean(psnr_values), 2)) + " dB")
        print("SSIM: " + str(round(np.mean(ssim_values), 4)))
        print("MAE:  " + str(round(np.mean(mae_values), 2)))

    print("\nRestored test images saved to:", RESTORED_DIR)


if __name__ == "__main__":
    main()