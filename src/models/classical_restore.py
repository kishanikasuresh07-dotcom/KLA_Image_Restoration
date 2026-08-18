"""
classical_restore.py

CLASSICAL (non-AI) restoration baseline.

Uses Non-Local Means Denoising (much stronger against Gaussian sensor
noise than a basic median filter) followed by light unsharp masking
to restore edge sharpness without re-amplifying noise.
"""

import os
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from metrics import evaluate_all

CLEAN_DIR = os.path.join("data", "clean")
DEGRADED_DIR = os.path.join("data", "degraded")
RESTORED_DIR = os.path.join("outputs", "restored", "classical")
os.makedirs(RESTORED_DIR, exist_ok=True)


def classical_restore(degraded_array):
    img_uint8 = degraded_array.astype(np.uint8)

    denoised = cv2.fastNlMeansDenoising(img_uint8, None, h=35, templateWindowSize=7, searchWindowSize=21)

    denoised_float = denoised.astype(np.float32)
    blurred = gaussian_filter(denoised_float, sigma=1.0)
    sharpened = denoised_float + 0.3 * (denoised_float - blurred)

    restored = np.clip(sharpened, 0, 255).astype(np.uint8)
    return restored


def main():
    degraded_files = [f for f in os.listdir(DEGRADED_DIR) if f.lower().endswith(".png")]

    if not degraded_files:
        print("No degraded images found in", DEGRADED_DIR)
        return

    all_results = []

    for filename in degraded_files:
        base_name = filename.split("_v")[0]
        clean_path = os.path.join(CLEAN_DIR, base_name + ".png")

        if not os.path.exists(clean_path):
            print("Skipping " + filename + ": matching clean image not found")
            continue

        degraded_img = np.array(Image.open(os.path.join(DEGRADED_DIR, filename)).convert("L"))
        clean_img = np.array(Image.open(clean_path).convert("L"))

        restored_img = classical_restore(degraded_img)

        out_path = os.path.join(RESTORED_DIR, filename)
        Image.fromarray(restored_img).save(out_path)

        metrics = evaluate_all(clean_img, restored_img)
        metrics["filename"] = filename
        all_results.append(metrics)

        print(filename + ": PSNR=" + str(round(metrics["PSNR"], 2)) + " dB, SSIM=" + str(round(metrics["SSIM"], 4)) + ", MAE=" + str(round(metrics["MAE"], 2)))

    if all_results:
        psnr_values = []
        ssim_values = []
        mae_values = []
        for r in all_results:
            psnr_values.append(r["PSNR"])
            ssim_values.append(r["SSIM"])
            mae_values.append(r["MAE"])

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_mae = np.mean(mae_values)

        print("\n--- AVERAGE (Classical Restoration) ---")
        print("PSNR: " + str(round(avg_psnr, 2)) + " dB")
        print("SSIM: " + str(round(avg_ssim, 4)))
        print("MAE:  " + str(round(avg_mae, 2)))

    print("\nRestored images saved to: " + RESTORED_DIR)


if __name__ == "__main__":
    main()