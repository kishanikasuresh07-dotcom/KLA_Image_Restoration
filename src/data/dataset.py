"""
dataset.py

Loads (clean, degraded) image pairs for training our CNN restoration model.

We split by IMAGE NAME (not by individual degraded variant) so that
the same underlying scene never appears in both the training set and
the test set. This avoids data leakage.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLEAN_DIR = os.path.join("data", "clean")
DEGRADED_DIR = os.path.join("data", "degraded")

IMG_SIZE = 256

TEST_IMAGE_NAMES = ["coins", "gravel"]


def load_and_resize(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


class RestorationDataset(Dataset):
    def __init__(self, split="train"):
        self.pairs = []

        degraded_files = [f for f in os.listdir(DEGRADED_DIR) if f.lower().endswith(".png")]

        for filename in degraded_files:
            base_name = filename.split("_v")[0]
            is_test_image = base_name in TEST_IMAGE_NAMES

            if split == "train" and is_test_image:
                continue
            if split == "test" and not is_test_image:
                continue

            clean_path = os.path.join(CLEAN_DIR, base_name + ".png")
            degraded_path = os.path.join(DEGRADED_DIR, filename)

            if os.path.exists(clean_path):
                self.pairs.append((degraded_path, clean_path, filename))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        degraded_path, clean_path, filename = self.pairs[idx]

        degraded_arr = load_and_resize(degraded_path)
        clean_arr = load_and_resize(clean_path)

        degraded_tensor = torch.from_numpy(degraded_arr).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_arr).unsqueeze(0)

        return degraded_tensor, clean_tensor, filename