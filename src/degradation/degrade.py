"""
degrade.py

Applies synthetic degradation to clean images to simulate a degraded
semiconductor-inspection-style image. This is a PROTOTYPE degradation
pipeline (not real KLA sensor data) built from three common, realistic
effects:

1. Gaussian noise    - simulates sensor/electronic noise
2. Gaussian blur      - simulates defocus/motion blur
3. Brightness/contrast reduction - simulates poor illumination

For each clean image, we generate several degraded variants with
different degradation strengths (so our model sees a range of
difficulty levels during training).

Degradation is reproducible: we use a fixed random seed per variant,
so re-running this script produces identical results.
"""

import os
import numpy as np
from PIL import Image
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

CLEAN_DIR = os.path.join("data", "clean")
DEGRADED_DIR = os.path.join("data", "degraded")
os.makedirs(DEGRADED_DIR, exist_ok=True)

# Each variant defines: noise strength, blur strength, brightness factor
# brightness factor < 1.0 means the image gets darker (illumination loss)
VARIANTS = [
    {"name": "v1_mild",      "noise_var": 0.002, "blur_sigma": 0.7, "brightness": 0.90, "seed": 1},
    {"name": "v2_moderate",  "noise_var": 0.005, "blur_sigma": 1.2, "brightness": 0.80, "seed": 2},
    {"name": "v3_strong",    "noise_var": 0.010, "blur_sigma": 1.8, "brightness": 0.70, "seed": 3},
    {"name": "v4_noisy",     "noise_var": 0.015, "blur_sigma": 0.5, "brightness": 0.95, "seed": 4},
    {"name": "v5_dark_blur", "noise_var": 0.004, "blur_sigma": 2.2, "brightness": 0.60, "seed": 5},
]


def degrade_image(clean_img_array, noise_var, blur_sigma, brightness, seed):
    """
    Takes a clean grayscale image (uint8 numpy array) and applies:
    brightness reduction -> blur -> noise, in that order.
    Returns a degraded uint8 numpy array.
    """
    # Step 1: work in float [0,1] for accurate math
    img = clean_img_array.astype(np.float32) / 255.0

    # Step 2: brightness/contrast reduction (illumination degradation)
    img = img * brightness

    # Step 3: blur (defocus/motion simulation)
    img = gaussian_filter(img, sigma=blur_sigma)

    # Step 4: add Gaussian noise (sensor noise simulation), reproducibly
    np.random.seed(seed)
    img = random_noise(img, mode="gaussian", var=noise_var, clip=True)

    # Step 5: convert back to uint8 [0,255]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def main():
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".png")]

    if not clean_files:
        print("No clean images found in", CLEAN_DIR)
        return

    print(f"Found {len(clean_files)} clean images. Generating {len(VARIANTS)} degraded variants each...\n")

    for filename in clean_files:
        base_name = os.path.splitext(filename)[0]
        clean_path = os.path.join(CLEAN_DIR, filename)
        clean_img = np.array(Image.open(clean_path).convert("L"))

        for variant in VARIANTS:
            degraded_array = degrade_image(
                clean_img,
                noise_var=variant["noise_var"],
                blur_sigma=variant["blur_sigma"],
                brightness=variant["brightness"],
                seed=variant["seed"],
            )
            out_name = f"{base_name}_{variant['name']}.png"
            out_path = os.path.join(DEGRADED_DIR, out_name)
            Image.fromarray(degraded_array).save(out_path)
            print(f"Saved: {out_path}")

    total = len(clean_files) * len(VARIANTS)
    print(f"\nDone! Generated {total} degraded images in {DEGRADED_DIR}")


if __name__ == "__main__":
    main()