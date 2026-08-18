"""
streamlit_app.py

Interactive demo for AI-Based Restoration of Degraded Images for
Semiconductor Inspection (KLA hackathon prototype).

Offers two restoration methods:
- Classical: fast (median/NLM denoising + sharpening)
- Deep Image Prior (DIP): slower (~1-2 min), optimizes a fresh
  neural network per image, no training dataset needed. Produces
  a stronger, more genuine AI-based restoration.

Run with:
    streamlit run app\\streamlit_app.py
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))

from classical_restore import classical_restore
from dip_restore import run_dip_restoration
from metrics import evaluate_all

st.set_page_config(page_title="KLA Image Restoration", layout="wide")

st.title("AI-Based Restoration of Degraded Images for Semiconductor Inspection")
st.caption("Hackathon prototype - KLA problem statement. Uses synthetic/prototype data, not real KLA datasets.")

st.markdown("---")

method = st.radio(
    "Choose restoration method",
    ["Deep Image Prior (AI, stronger, ~1-2 min)", "Classical (fast, ~1 sec)"],
    index=0
)

uploaded_file = st.file_uploader("Upload a degraded inspection-like image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("L")
    original_array = np.array(original_img).astype(np.float32)/255.0

    if method.startswith("Deep Image Prior"):
        st.info("Running Deep Image Prior optimization - this takes about 1-2 minutes since it trains a small network specifically for this image, with no pre-existing dataset.")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total):
            if current%5 == 0 or current == total:
                progress_bar.progress(current / total)
                status_text.text("Optimizing: iteration " + str(current) + " / " + str(total))

        restored_array = run_dip_restoration(original_array, iterations=220, progress_callback=update_progress)
        st.write("Min:", restored_array.min(), "Max:", restored_array.max(), "Mean:", restored_array.mean())
        status_text.text("Done!")
    else:
        with st.spinner("Applying classical restoration..."):
            restored_array = classical_restore(original_array)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original (Degraded)")
        st.image(original_array, use_container_width=True, clamp=True)

    with col2:
        st.subheader("Restored")
        st.image(restored_array, use_container_width=True, clamp=True)

    st.markdown("---")
    st.subheader("Quality Metrics")

    reference_file = st.file_uploader(
        "Optional: upload the matching clean/reference image to compute PSNR/SSIM/MAE",
        type=["png", "jpg", "jpeg"],
        key="reference"
    )

    if reference_file is not None:
        reference_img = Image.open(reference_file).convert("L")
        reference_array = np.array(reference_img.resize((original_array.shape[1], original_array.shape[0])))

        metrics = evaluate_all(reference_array, restored_array)

        m1, m2, m3 = st.columns(3)
        m1.metric("PSNR (dB)", str(round(metrics["PSNR"], 2)))
        m2.metric("SSIM", str(round(metrics["SSIM"], 4)))
        m3.metric("MAE", str(round(metrics["MAE"], 2)))
    else:
        st.info("Metrics requiring a reference image (PSNR/SSIM/MAE) cannot be calculated without one. Upload a clean reference above if available.")

else:
    st.info("Upload an image above to get started. You can use any of the sample degraded images from data/degraded/ in this project as a demo.")

st.markdown("---")
st.caption("Prototype built for a student hackathon. Two restoration methods available: Deep Image Prior (per-image optimization, no training data needed) and classical filtering (Non-Local Means denoising + unsharp masking). A CNN/U-Net model was also trained and evaluated separately - see project README.")