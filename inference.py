"""
inference.py

Simple command-line restoration tool.

Usage:
    python inference.py --input path\to\image.png

Loads an image, applies classical restoration (denoise + sharpen),
and saves the result to outputs/restored/inference_result.png
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.append(os.path.join("src", "models"))
from classical_restore import classical_restore

OUTPUT_DIR = os.path.join("outputs", "restored")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Restore a degraded image.")
    parser.add_argument("--input", required=True, help="Path to the input image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Error: input file not found:", args.input)
        return

    print("Loading image:", args.input)
    img = Image.open(args.input).convert("L")
    img_array = np.array(img)

    print("Applying restoration...")
    restored_array = classical_restore(img_array)

    out_path = os.path.join(OUTPUT_DIR, "inference_result.png")
    Image.fromarray(restored_array).save(out_path)

    print("Done!")
    print("Restored image saved to:", out_path)


if __name__ == "__main__":
    main()