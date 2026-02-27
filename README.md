# YOLO11n Raspberry Pi 4 Benchmark — Complete Guide

## Overview

This guide walks through benchmarking the **YOLO11n** (nano) object detection model on a **Raspberry Pi 4** using images from the COCO 2017 dataset. The benchmark measures inference speed, memory usage, CPU temperature, and detection quality at two resolutions (320×320 and 640×640).

**Why YOLO11n?** It's the smallest model in the YOLO11 family (~2.6M parameters), specifically designed for edge devices with limited compute.

**Why NCNN export?** On ARM devices like the Pi, NCNN runs significantly faster than PyTorch because it avoids the heavy Python/CUDA runtime overhead and is optimized for ARM NEON instructions.

---

## Prerequisites

- Raspberry Pi 4 (2GB+ RAM recommended)
- Raspberry Pi OS (64-bit) — **important: use 64-bit for best performance**
- Miniforge/conda installed
- A conda environment named `yolo_demo`
- Internet connection (for downloading COCO images)
- Adequate storage (~1GB free for model, images, and dependencies)
- **A separate machine** (PC, laptop, or Google Colab) for the one-time model export

> **Why export elsewhere?** Recent PyTorch wheels use CPU instructions (e.g. SVE) that the Pi 4's Cortex-A72 doesn't support, causing "Illegal instruction" crashes. We avoid this by exporting the model on a capable machine and only running the lightweight NCNN inference on the Pi.

---

## File Structure

```
yolo_benchmark/
├── 01_setup.sh                  # Install dependencies
├── 02_download_coco_sample.py   # Download 100 COCO images
├── 03_benchmark.py              # Run the benchmark
├── README.md                    # This guide
├── yolo11n_ncnn_model/          # (copied from export machine) NCNN model
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

The Pi 4's Cortex-A72 can't run recent PyTorch wheels, so we export on a separate machine:

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=ncnn
```

This creates a `yolo11n_ncnn_model/` folder. Copy it to your Pi:

```bash
scp -r yolo11n_ncnn_model/ pi@<pi-ip>:~/yolo_benchmark/
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
| `ultralytics` | YOLO11 framework (model loading + NCNN inference) |
| `opencv-python-headless` | Image loading and processing |
| `ncnn` | NCNN inference runtime for ARM |
| `psutil` | System monitoring (RAM, CPU temp, CPU usage) |
| `numpy<2.0` | Pi-compatible NumPy (v2.0+ can cause illegal instruction) |

**Expected time:** 5–15 minutes depending on your internet speed and SD card.

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

> **Note:** The full COCO val2017 set is 6GB+. This script avoids that by downloading only the images we need.

### Step 5: Run the Benchmark

```bash
python 03_benchmark.py
```

This script:
1. Collects system information (Pi model, RAM, CPU, etc.)
2. Downloads YOLO11n weights (if not already present)
3. Exports the model from PyTorch to NCNN format
4. Runs 5 warmup inferences (to stabilize timing)
5. Benchmarks all 100 images at **320×320**
6. Benchmarks all 100 images at **640×640**
7. Generates a Markdown report and JSON data files

**Expected time:** 10–30 minutes total (depends on your Pi and cooling).

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
- **Mean / Median / Min / Max** inference time per image (milliseconds)
- **P95 / P99** latency percentiles
- **FPS** (frames per second)
- **Total** wall-clock time for all 100 images

### Memory
- **Model load delta:** How much RAM the model adds when loaded
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

These are ballpark figures for a Pi 4 (4GB) with NCNN export:

| Metric | 320×320 | 640×640 |
|--------|---------|---------|
| Mean inference | ~150–250ms | ~500–1000ms |
| FPS | ~4–7 | ~1–2 |
| Peak RAM | ~300–500MB | ~500–800MB |
| CPU temp rise | ~5–15°C | ~10–20°C |

Your results will vary based on cooling, overclock settings, background processes, and ambient temperature.

---

## Troubleshooting

### "Killed" or out-of-memory errors
- Close other applications (especially browsers)
- Run `free -h` to check available RAM before starting
- If still failing at 640, try running only the 320 benchmark by editing `IMAGE_SIZES` in `03_benchmark.py`

### NCNN model fails to load
- Ensure the `yolo11n_ncnn_model/` folder contains the model files (`.bin` and `.param` files)
- Make sure the model was exported with the same ultralytics version as installed on the Pi
- If you see "Illegal instruction" even with NCNN, check NumPy: `pip install "numpy<2.0"`

### Very slow inference
- Make sure you're on **64-bit** Raspberry Pi OS (`uname -m` should show `aarch64`)
- Check CPU throttling: `vcgencmd get_throttled` (0x0 = no throttling)
- Ensure adequate cooling — the Pi will throttle at 80°C+

### Temperature warnings
- Add a heatsink and fan if you haven't already
- Run `vcgencmd measure_temp` to monitor temperature
- Consider active cooling for sustained workloads

---

## Customization

You can modify `03_benchmark.py` to adjust:

```python
IMAGE_SIZES = [320, 640]    # Add/remove resolutions (e.g., [256, 320, 480, 640])
WARMUP_RUNS = 5             # Increase for more stable timing
NCNN_MODEL_DIR = "yolo11n_ncnn_model"  # Path to your pre-exported model
```

To benchmark a different model, export it on your PC first:
```bash
# On your PC:
yolo export model=yolo11s.pt format=ncnn
# Then scp the folder to the Pi and update NCNN_MODEL_DIR
```

---

## Next Steps

- **Compare models:** Run with `yolo11s` or `yolo11m` to see the speed/accuracy trade-off
- **Real-time camera:** Try `picamera2` + YOLO for live detection
- **Optimize further:** Overclock the Pi, use a Pi 5, or try quantized models
- **Accuracy evaluation:** Use the COCO annotations to compute mAP scores (not just speed)