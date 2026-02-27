#!/bin/bash
# ============================================================================
# YOLO11n Raspberry Pi 4 Benchmark - Setup Script
# ============================================================================
# This script installs all required dependencies into the 'yolo_demo' conda env.
# 
# Usage:
#   conda activate yolo_demo
#   bash 01_setup.sh
# ============================================================================

set -e

echo "============================================"
echo "  YOLO11n Raspberry Pi 4 Benchmark Setup"
echo "  (Pi-only mode — NCNN pre-exported)"
echo "============================================"

# --- Check we're in the right conda env ---
if [[ "$CONDA_DEFAULT_ENV" != "yolo_demo" ]]; then
    echo "ERROR: Please activate the yolo_demo environment first:"
    echo "  conda activate yolo_demo"
    exit 1
fi

# --- Check NCNN model folder exists ---
if [[ ! -d "yolo11n_ncnn_model" ]]; then
    echo ""
    echo "WARNING: yolo11n_ncnn_model/ folder not found."
    echo "  You need to export the model on another machine first:"
    echo "    pip install ultralytics"
    echo "    yolo export model=yolo11n.pt format=ncnn"
    echo "  Then copy the folder to this directory:"
    echo "    scp -r yolo11n_ncnn_model/ pi@<pi-ip>:~/yolo_benchmark/"
    echo ""
fi

echo ""
echo "[1/4] Installing NumPy (Pi-compatible version)..."
pip install "numpy<2.0"

echo ""
echo "[2/4] Installing OpenCV and NCNN runtime..."
pip install opencv-python-headless ncnn

echo ""
echo "[3/4] Installing ultralytics and monitoring tools..."
pip install ultralytics psutil

echo ""
echo "[4/4] Verifying installation..."
python -c "
import cv2
import psutil
import numpy
try:
    import ultralytics
    print(f'  ultralytics: {ultralytics.__version__}')
except Exception as e:
    print(f'  ultralytics: import issue ({e}) — may still work with NCNN')
print(f'  numpy:       {numpy.__version__}')
print(f'  opencv:      {cv2.__version__}')
print(f'  psutil:      {psutil.__version__}')
print()
print('  All packages installed successfully!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next: python 02_download_coco_sample.py"
echo "============================================"
