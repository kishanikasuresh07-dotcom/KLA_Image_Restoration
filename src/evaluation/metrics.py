"""
metrics.py

Reusable image-quality metric functions used throughout this project
to evaluate restoration quality.

PSNR (Peak Signal-to-Noise Ratio): measures pixel-level similarity.
Higher is better. Measured in decibels (dB). It tells us how much
"error" (in signal terms) exists between two images.

SSIM (Structural Similarity Index): measures similarity in local
structure/texture/contrast, closer to how humans perceive image
quality. Ranges from -1 to 1, where 1 means identical images.

MAE (Mean Absolute Error): the average absolute difference between
pixel values. Lower is better. Simple and easy to interpret
(e.g. MAE of 5 means pixels are off by 5 out of 255 on average).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(clean_img, restored_img):
    """Both images must be numpy arrays of the same shape, dtype uint8."""
    return peak_signal_noise_ratio(clean_img, restored_img, data_range=255)


def compute_ssim(clean_img, restored_img):
    """Both images must be numpy arrays of the same shape, dtype uint8."""
    return structural_similarity(clean_img, restored_img, data_range=255)


def compute_mae(clean_img, restored_img):
    """Both images must be numpy arrays of the same shape."""
    return np.mean(np.abs(clean_img.astype(np.float32) - restored_img.astype(np.float32)))


def evaluate_all(clean_img, restored_img):
    """Returns a dictionary with PSNR, SSIM, and MAE."""
    return {
        "PSNR": compute_psnr(clean_img, restored_img),
        "SSIM": compute_ssim(clean_img, restored_img),
        "MAE": compute_mae(clean_img, restored_img),
    }