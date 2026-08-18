import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

IMG_SIZE = 256
DEFAULT_INFERENCE_ITERATIONS = 120


def load_and_resize(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


class DIPNet(nn.Module):
    def __init__(self, in_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return torch.sigmoid(x)


def run_dip_restoration(degraded_array, iterations=DEFAULT_INFERENCE_ITERATIONS, progress_callback=None):
    """
    Runs DIP restoration on a single degraded image array (any size, grayscale).
    Returns the restored image as a NumPy array (float32, 0-1 range).
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = degraded_array.shape

    print(f"[DEBUG] Input min={degraded_array.min():.4f} max={degraded_array.max():.4f} mean={degraded_array.mean():.4f}")
    if degraded_array.max() > 1.0:
       degraded_array = degraded_array / 255.0

    target = torch.from_numpy(degraded_array).float().unsqueeze(0).unsqueeze(0).to(device)
    net_input = torch.rand(1, 32, h, w, device=device) * 0.1  # small-scale noise input

    model = DIPNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # lowered from 0.01
    criterion = nn.MSELoss()

    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(net_input)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent explosion
        optimizer.step()

        if i % 20 == 0 or i == iterations - 1:
            print(f"[DEBUG] Iter {i}: loss={loss.item():.6f}  out_min={output.min().item():.4f}  out_max={output.max().item():.4f}")

        if progress_callback is not None:
            progress_callback(i + 1, iterations)

    model.eval()
    with torch.inference_model():
        restored = model(net_input).squeeze().cpu().numpy()

    restored = np.clip(restored, 0.0, 1.0).astype(np.float32)

    print(f"[DEBUG] Output min={restored.min():.4f} max={restored.max():.4f} mean={restored.mean():.4f}")

    return restored