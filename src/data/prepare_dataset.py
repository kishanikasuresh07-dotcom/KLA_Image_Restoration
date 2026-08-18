"""
prepare_dataset.py

This script saves a set of built-in scikit-image sample images
into data/clean/ as PNG files. These act as our "clean" ground-truth
images for this hackathon prototype (a stand-in for real semiconductor
inspection images, since we don't have access to real KLA data).
"""

import os
from skimage import data
from PIL import Image
import numpy as np

# Folder where clean images will be saved
OUTPUT_DIR = os.path.join("data", "clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary of sample images: name -> image array
sample_images = {
    "camera": data.camera(),
    "brick": data.brick(),
    "grass": data.grass(),
    "checkerboard": data.checkerboard(),
    "coins": data.coins(),
    "gravel": data.gravel(),
}

print("Saving clean sample images...")

for name, img_array in sample_images.items():
    # Ensure image is in 0-255 uint8 format
    img_array = img_array.astype(np.uint8)

    # Convert numpy array to a PIL Image (grayscale mode "L")
    img = Image.fromarray(img_array, mode="L")

    save_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    img.save(save_path)
    print(f"Saved: {save_path}  (shape: {img_array.shape})")

print("\nDone! All clean images saved to:", OUTPUT_DIR)