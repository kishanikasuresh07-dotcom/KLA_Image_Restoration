# KLA Image Restoration

AI-based restoration of degraded images for semiconductor wafer inspection, built for the KLA hackathon challenge.

## Overview

Semiconductor wafer inspection images can suffer from noise, blur, and other degradations that make defect detection harder. This project restores degraded wafer images using multiple approaches — a classical baseline and a Deep Image Prior (DIP) deep learning method — and provides an interactive Streamlit demo to visualize results with live quality metrics (PSNR, SSIM, MAE).

## Features

- Synthetic degradation pipeline to simulate real-world image defects
- Classical restoration baseline (Non-Local Means)
- CNN/U-Net based restoration
- Deep Image Prior (DIP) restoration using an untrained hourglass encoder-decoder network
- Streamlit web app for interactive upload, restore, compare workflow
- Live quality metrics: PSNR, SSIM, MAE
- Visual comparison outputs and showcase strips for presentation

## Project Structure

KLA_Image_Restoration/
- app/
  - streamlit_app.py (Streamlit demo application)
- src/
  - data/ (clean, degraded, raw, train/val/test)
  - degradation/
    - degrade.py (image degradation pipeline)
  - models/
    - classical_restore.py
    - dip_restore.py (Deep Image Prior restoration)
    - unet.py (U-Net model)
  - training/ (model training scripts)
  - evaluation/ (metrics and evaluation)
  - inference/ (inference utilities)
- outputs/
  - comparisons/ (side-by-side result images)
  - metrics/
  - plots/
  - restored/
- inference.py
- make_dip_showcase.py
- requirements.txt
- README.md

## Installation

1. Clone this repository:

git clone https://github.com/kishanikasuresh07-dotcom/KLA_Image_Restoration.git
cd KLA_Image_Restoration

2. Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

## Usage

To run the Streamlit demo:

streamlit run app/streamlit_app.py

This will open the app in your browser, where you can upload a degraded image and view the restored output along with quality metrics.

## Results

Sample restoration comparisons are available in outputs/comparisons/, including a showcase strip demonstrating the DIP restoration process.

## Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy / OpenCV

## Acknowledgements

Built for the KLA hackathon challenge on AI-based restoration of degraded images for semiconductor inspection.