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
echo "============================================"

# --- Check we're in the right conda env ---
if [[ "$CONDA_DEFAULT_ENV" != "yolo_demo" ]]; then
    echo "ERROR: Please activate the yolo_demo environment first:"
    echo "  conda activate yolo_demo"
    exit 1
fi

echo ""
echo "[1/4] Installing core Python packages..."
pip install ultralytics

echo ""
echo "[2/4] Installing benchmarking & monitoring tools..."
pip install psutil

echo ""
echo "[3/4] Installing COCO dataset tools..."
pip install pycocotools

echo ""
echo "[4/4] Verifying installation..."
python -c "
import ultralytics
import psutil
print(f'  ultralytics: {ultralytics.__version__}')
print(f'  psutil:      {psutil.__version__}')
print('  All packages installed successfully!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next: python 02_download_coco_sample.py"
echo "============================================"
