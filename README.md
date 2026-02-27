# YOLO11n Raspberry Pi 4 Benchmark — Complete Guide

## Overview

This guide walks through benchmarking the **YOLO11n** (nano) object detection model on a **Raspberry Pi 4** using images from the COCO 2017 dataset. The benchmark measures inference speed, memory usage, CPU temperature, and detection quality at two resolutions (320×320 and 640×640).

**Why YOLO11n?** It's the smallest model in the YOLO11 family (~2.6M parameters), designed for edge devices with limited compute.

**Why ONNX Runtime?** Recent PyTorch wheels use CPU instructions (SVE, etc.) that the Pi 4's Cortex-A72 (ARMv8.0) doesn't support, causing "Illegal instruction" crashes. ONNX Runtime has proper aarch64 wheels that work reliably on the Pi 4, and runs inference efficiently without needing PyTorch at all.

---

## Prerequisites

- Raspberry Pi 4 (2GB+ RAM recommended)
- Raspberry Pi OS (64-bit) — **important: use 64-bit for best performance**
- Miniforge/conda installed
- A conda environment named `yolo_demo`
- Internet connection (for downloading COCO images)
- Adequate storage (~1GB free for model, images, and dependencies)
- **A separate machine** (PC, laptop, or Google Colab) for the one-time model export

---

## File Structure

```
yolo_benchmark/
├── 01_setup.sh                  # Install dependencies (torch-free)
├── 02_download_coco_sample.py   # Download 100 COCO images
├── 03_benchmark.py              # Run the benchmark (ONNX Runtime)
├── README.md                    # This guide
├── yolo11n.onnx                 # (copied from export machine) ONNX model
├── coco_sample/                 # (created) Downloaded images
│   ├── manifest.json            # Image metadata for reproducibility
│   └── *.jpg                    # 100 COCO validation images
└── results/                     # (created) Benchmark outputs
    ├── BENCHMARK_REPORT.md      # Human-readable report
    ├── benchmark_combined.json  # Full combined results
    ├── benchmark_320.json       # Raw results for 320x320
    └── benchmark_640.json       # Raw results for 640x640
```

---

## Step-by-Step Instructions

### Step 1: Export the Model (on your PC/laptop, NOT the Pi)

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx
```

This creates `yolo11n.onnx`. Copy it to your Pi:

```bash
scp yolo11n.onnx pi@<pi-ip>:~/yolo_benchmark/
```

Alternatively, use [Google Colab](https://colab.research.google.com) if you don't have a suitable local machine.

### Step 2: Activate Your Environment (on the Pi)

```bash
conda activate yolo_demo
```

### Step 3: Install Dependencies

```bash
cd ~/yolo_benchmark
bash 01_setup.sh
```

**What gets installed:**

| Package | Purpose |
|---------|---------|
| `onnxruntime` | Model inference (replaces PyTorch entirely) |
| `opencv-python-headless` | Image loading, preprocessing, NMS |
| `numpy<2.0` | Pi-compatible NumPy (v2.0+ causes illegal instruction) |
| `psutil` | System monitoring (RAM, CPU temp, CPU usage) |

**What is NOT needed:** PyTorch, ultralytics, CUDA, torchvision. The setup script actively uninstalls these if present to save space and avoid conflicts.

**Expected time:** 3–10 minutes.

### Step 4: Download COCO Sample Images

```bash
python 02_download_coco_sample.py
```

This script:
1. Downloads the COCO 2017 validation annotations (~252MB zip)
2. Randomly selects 100 images (seed=42 for reproducibility)
3. Downloads only those 100 images individually (~15MB total)
4. Saves a manifest.json for reproducibility

**Expected time:** 2–5 minutes.

### Step 5: Run the Benchmark

```bash
python 03_benchmark.py
```

This script:
1. Collects system information (Pi model, RAM, CPU, etc.)
2. Loads the ONNX model into ONNX Runtime
3. Runs 5 warmup inferences (to stabilize timing)
4. Benchmarks all 100 images at **320×320**
5. Benchmarks all 100 images at **640×640**
6. Generates a Markdown report and JSON data files

**Expected time:** 5–20 minutes total.

### Step 6: View Results

```bash
cat results/BENCHMARK_REPORT.md
```

Or copy the report to your computer:
```bash
# From your computer (not the Pi):
scp pi@<pi-ip>:~/yolo_benchmark/results/BENCHMARK_REPORT.md .
```

---

## What Gets Measured

### Inference Timing
The benchmark separates timing into three stages:

- **Preprocess:** Letterbox resize + normalize + format conversion
- **Inference:** ONNX Runtime model execution (the core measurement)
- **Postprocess:** Confidence filtering + NMS

Each stage is timed independently, plus full pipeline FPS.

### Memory
- **Peak process RSS:** Maximum resident set size during inference
- **System RAM:** Available memory before and after the benchmark

### Thermal & Power
- **CPU temperature** before and after each benchmark run
- **Temperature delta** as a proxy for thermal/power load

> **Important:** True power consumption (watts) cannot be measured via software on the Pi. For accurate power measurement, use an external USB-C inline power meter between your power supply and the Pi. Typical Pi 4 power consumption is ~3W idle, ~6-7W under full CPU load.

### Detection Quality
- Total number of detections across all images
- Mean and max detections per image
- This helps verify the model is working correctly at both resolutions

---

## Expected Results (Rough Estimates)

Ballpark figures for a Pi 4 (4GB) with ONNX Runtime:

| Metric | 320×320 | 640×640 |
|--------|---------|---------|
| Mean inference | ~100–200ms | ~400–800ms |
| Pipeline FPS | ~4–8 | ~1–2 |
| Peak RAM | ~200–400MB | ~300–600MB |
| CPU temp rise | ~5–15°C | ~10–20°C |

Results vary based on cooling, overclock settings, background processes, and ambient temperature. ONNX Runtime is generally faster than NCNN for this model on Pi 4.

---

## Troubleshooting

### "Illegal instruction" on import
- This usually means `numpy>=2.0` was installed. Fix: `pip install "numpy<2.0"`
- Verify: `python -c "import numpy; import cv2; import onnxruntime; print('OK')"`

### Out of memory / "Killed"
- Close other applications (especially browsers)
- Run `free -h` to check available RAM before starting
- If still failing at 640, edit `IMAGE_SIZES = [320]` in `03_benchmark.py`

### ONNX model fails to load
- Ensure `yolo11n.onnx` was exported correctly (should be ~10MB)
- Try re-exporting with a specific opset: `yolo export model=yolo11n.pt format=onnx opset=12`

### Very slow inference
- Make sure you're on **64-bit** Raspberry Pi OS (`uname -m` → `aarch64`)
- Check CPU throttling: `vcgencmd get_throttled` (0x0 = no throttling)
- Ensure adequate cooling — the Pi will throttle at 80°C+

### Zero detections
- This is normal for some images at 320×320 with the confidence threshold of 0.25
- If you get zero across ALL images, the model may not have exported correctly

---

## Customization

Edit the configuration at the top of `03_benchmark.py`:

```python
IMAGE_SIZES = [320, 640]     # Add/remove resolutions (e.g., [256, 320, 480, 640])
WARMUP_RUNS = 5              # Increase for more stable timing
CONF_THRESHOLD = 0.25        # Lower = more detections, higher = fewer
IOU_THRESHOLD = 0.45         # NMS overlap threshold
ONNX_MODEL_PATH = "yolo11n.onnx"
```

To benchmark a different model, export it on your PC:
```bash
yolo export model=yolo11s.pt format=onnx    # Small (~9.4M params)
yolo export model=yolo11m.pt format=onnx    # Medium (~25.3M params)
```
Then copy the `.onnx` file to the Pi and update `ONNX_MODEL_PATH`.

---

## Next Steps

- **Compare models:** Export and benchmark `yolo11s` or `yolo11m` for speed/accuracy trade-offs
- **Real-time camera:** Use `picamera2` + this inference pipeline for live detection
- **Thread tuning:** Try different `intra_op_num_threads` values (1, 2, 4) to find the optimum
- **Accuracy evaluation:** Use COCO annotations to compute mAP scores
- **Quantization:** Export with INT8 quantization for potential speed gains