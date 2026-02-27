#!/bin/bash
# ============================================================================
# YOLO11n Raspberry Pi 4 Benchmark - Setup Script
# ============================================================================
# Torch-free pipeline: uses ONNX Runtime for inference.
#
# Usage:
#   conda activate yolo_demo
#   bash 01_setup.sh
# ============================================================================

set -e

echo "============================================"
echo "  YOLO11n Raspberry Pi 4 Benchmark Setup"
echo "  (Torch-free — ONNX Runtime only)"
echo "============================================"

# --- Check we're in the right conda env ---
if [[ "$CONDA_DEFAULT_ENV" != "yolo_demo" ]]; then
    echo "ERROR: Please activate the yolo_demo environment first:"
    echo "  conda activate yolo_demo"
    exit 1
fi

# --- Ensure conda env has its own pip ---
echo ""
echo "[0/3] Ensuring conda env has its own pip..."
conda install -y pip

# --- Check ONNX model exists ---
if [[ ! -f "yolo11n.onnx" ]]; then
    echo ""
    echo "WARNING: yolo11n.onnx not found in this directory."
    echo "  Export on another machine first:"
    echo "    pip install ultralytics"
    echo "    yolo export model=yolo11n.pt format=onnx"
    echo "  Then copy to the Pi:"
    echo "    scp yolo11n.onnx pi@<pi-ip>:~/yolo_benchmark/"
    echo ""
fi

echo ""
echo "[1/3] Installing core packages..."
pip install "numpy<2.0" opencv-python-headless onnxruntime psutil

echo ""
echo "[2/3] Uninstalling torch/ultralytics if present (not needed)..."
pip uninstall -y torch torchvision ultralytics 2>/dev/null || true

echo ""
echo "[3/3] Verifying installation..."
python -c "
import numpy
import cv2
import onnxruntime as ort
import psutil

print(f'  numpy:        {numpy.__version__}')
print(f'  opencv:       {cv2.__version__}')
print(f'  onnxruntime:  {ort.__version__}')
print(f'  psutil:       {psutil.__version__}')
print(f'  ORT providers: {ort.get_available_providers()}')
print()
print('  All packages installed successfully!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next: python 02_download_coco_sample.py"
echo "============================================"
