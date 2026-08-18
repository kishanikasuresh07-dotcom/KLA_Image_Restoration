"""
make_dip_showcase.py

Combines Degraded, DIP-early, DIP-best, and Clean images into one
side-by-side comparison strip for presentation slides.
"""

from PIL import Image, ImageDraw, ImageFont
import os

IMG_SIZE = 256
LABEL_HEIGHT = 40

files_and_labels = [
    (os.path.join("data", "degraded", "camera_v3_strong.png"), "Degraded Input"),
    (os.path.join("outputs", "restored", "dip", "iter_0025.png"), "DIP - Iteration 25"),
    (os.path.join("outputs", "restored", "dip", "iter_0225.png"), "DIP - Iteration 225 (Best)"),
    (os.path.join("data", "clean", "camera.png"), "Original Clean"),
]

images = []
for path, label in files_and_labels:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    images.append((img, label))

strip_width = IMG_SIZE * len(images)
strip_height = IMG_SIZE + LABEL_HEIGHT

strip = Image.new("RGB", (strip_width, strip_height), color="white")
draw = ImageDraw.Draw(strip)

for i, (img, label) in enumerate(images):
    x_offset = i * IMG_SIZE
    strip.paste(img.convert("RGB"), (x_offset, LABEL_HEIGHT))
    text_width = draw.textlength(label)
    text_x = x_offset + (IMG_SIZE - text_width) // 2
    draw.text((text_x, 10), label, fill="black")

out_dir = os.path.join("outputs", "comparisons")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "dip_showcase_strip.png")
strip.save(out_path)

print("Showcase strip saved to:", out_path)