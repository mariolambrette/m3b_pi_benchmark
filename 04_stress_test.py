#!/usr/bin/env python3
"""
04_stress_test.py
Long-running thermal and stability stress test for YOLO11n on Raspberry Pi 4.

Runs continuous inference loops at 320x320 and 640x640, sampling system
metrics (RAM %, CPU temperature) every minute. Includes a 1-hour cooldown
between resolutions.

Timeline:
  - 3 hours: continuous inference at 320x320
  - 1 hour:  cooldown (idle)
  - 3 hours: continuous inference at 640x640
  - Total:   ~7 hours

Usage:
    python 04_stress_test.py

Outputs saved to results/stress_test/:
    stress_320.json      — per-minute samples for 320x320
    stress_640.json      — per-minute samples for 640x640
    cooldown.json        — per-minute samples during cooldown
    stress_summary.json  — overall summary
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import onnxruntime as ort
import psutil

# ── Configuration ──────────────────────────────────────────────────────────
ONNX_MODEL_PATH = "yolo11n.onnx"
IMAGE_DIR = "coco_sample"
RESULTS_DIR = os.path.join("results", "stress_test")

INFERENCE_DURATION_HOURS = 3
COOLDOWN_DURATION_HOURS = 1
SAMPLE_INTERVAL_SECONDS = 60    # Sample metrics every minute

IMAGE_SIZES = [320, 640]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ── YOLO Pre/Post Processing (same as 03_benchmark.py) ────────────────────

def letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img


def preprocess(img_bgr, imgsz):
    img = letterbox(img_bgr, new_shape=(imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    preds = output[0].squeeze(0).T
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    max_scores = np.max(class_scores, axis=1)
    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]

    if len(boxes_cxcywh) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes_cxcywh.tolist(), max_scores.tolist(),
        conf_thresh, iou_thresh,
    )
    return indices.flatten().tolist() if len(indices) > 0 else []


# ── System Monitoring ──────────────────────────────────────────────────────

def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            return round(temps["cpu_thermal"][0].current, 1)
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def get_sample():
    """Take a single metrics sample."""
    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": None,  # filled in by caller
        "cpu_temp_c": get_cpu_temperature(),
        "ram_used_percent": round(vm.percent, 1),
        "ram_available_mb": round(vm.available / (1024**2), 1),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "process_rss_mb": round(
            psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1
        ),
    }


# ── Inference Loop ─────────────────────────────────────────────────────────

def run_inference_phase(session, input_name, images, imgsz, duration_hours):
    """
    Run continuous inference for the specified duration.
    Samples system metrics every SAMPLE_INTERVAL_SECONDS.

    Returns:
        samples: list of per-minute metric dicts
        total_inferences: total number of inferences performed
    """
    duration_s = duration_hours * 3600
    end_time = time.monotonic() + duration_s

    samples = []
    total_inferences = 0
    inference_times_ms = []   # rolling window for per-sample averages
    num_images = len(images)
    img_index = 0

    # Take initial sample
    psutil.cpu_percent(interval=None)  # reset counter
    start_time = time.monotonic()
    last_sample_time = start_time
    sample_number = 0

    initial_sample = get_sample()
    initial_sample["elapsed_minutes"] = 0
    initial_sample["inferences_since_last"] = 0
    initial_sample["mean_inference_ms"] = 0
    initial_sample["total_inferences"] = 0
    samples.append(initial_sample)

    print(f"  Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  End time:   {(datetime.now() + timedelta(hours=duration_hours)).strftime('%H:%M:%S')}")
    print(f"  Sampling every {SAMPLE_INTERVAL_SECONDS}s")
    print()

    while time.monotonic() < end_time:
        # Run one inference
        blob = preprocess(images[img_index], imgsz)

        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})
        t1 = time.perf_counter()

        postprocess(output)

        inference_ms = (t1 - t0) * 1000
        inference_times_ms.append(inference_ms)
        total_inferences += 1
        img_index = (img_index + 1) % num_images

        # Check if it's time to take a sample
        now = time.monotonic()
        if now - last_sample_time >= SAMPLE_INTERVAL_SECONDS:
            sample_number += 1
            elapsed_min = round((now - start_time) / 60, 1)

            sample = get_sample()
            sample["elapsed_minutes"] = elapsed_min
            sample["inferences_since_last"] = len(inference_times_ms)
            sample["mean_inference_ms"] = round(
                sum(inference_times_ms) / len(inference_times_ms), 2
            ) if inference_times_ms else 0
            sample["total_inferences"] = total_inferences
            samples.append(sample)

            # Print progress
            remaining_min = round((end_time - now) / 60, 0)
            temp_str = f"{sample['cpu_temp_c']}°C" if sample['cpu_temp_c'] else "N/A"
            print(
                f"  [{elapsed_min:>6.1f} min] "
                f"Temp: {temp_str:>7s} | "
                f"RAM: {sample['ram_used_percent']:>5.1f}% | "
                f"Inf/min: {sample['inferences_since_last']:>4d} | "
                f"Avg: {sample['mean_inference_ms']:>7.1f}ms | "
                f"Remaining: {remaining_min:.0f} min"
            )

            # Reset rolling window
            inference_times_ms = []
            last_sample_time = now

    # Final sample
    final_elapsed = round((time.monotonic() - start_time) / 60, 1)
    final_sample = get_sample()
    final_sample["elapsed_minutes"] = final_elapsed
    final_sample["inferences_since_last"] = len(inference_times_ms)
    final_sample["mean_inference_ms"] = round(
        sum(inference_times_ms) / len(inference_times_ms), 2
    ) if inference_times_ms else 0
    final_sample["total_inferences"] = total_inferences
    samples.append(final_sample)

    return samples, total_inferences


def run_cooldown(duration_hours):
    """
    Idle for the specified duration, sampling metrics every minute.
    Returns list of per-minute metric samples.
    """
    duration_s = duration_hours * 3600
    end_time = time.monotonic() + duration_s
    start_time = time.monotonic()

    samples = []
    psutil.cpu_percent(interval=None)

    initial = get_sample()
    initial["elapsed_minutes"] = 0
    samples.append(initial)

    print(f"  Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Resume at:  {(datetime.now() + timedelta(hours=duration_hours)).strftime('%H:%M:%S')}")
    print()

    while time.monotonic() < end_time:
        time.sleep(SAMPLE_INTERVAL_SECONDS)

        elapsed_min = round((time.monotonic() - start_time) / 60, 1)
        sample = get_sample()
        sample["elapsed_minutes"] = elapsed_min
        samples.append(sample)

        remaining_min = round((end_time - time.monotonic()) / 60, 0)
        temp_str = f"{sample['cpu_temp_c']}°C" if sample['cpu_temp_c'] else "N/A"
        print(
            f"  [{elapsed_min:>6.1f} min] "
            f"Temp: {temp_str:>7s} | "
            f"RAM: {sample['ram_used_percent']:>5.1f}% | "
            f"Remaining: {remaining_min:.0f} min"
        )

    return samples


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    total_hours = (INFERENCE_DURATION_HOURS * 2) + COOLDOWN_DURATION_HOURS
    print("=" * 60)
    print("  YOLO11n Raspberry Pi 4 Stress Test")
    print(f"  Total estimated time: ~{total_hours} hours")
    print("=" * 60)
    print(f"\n  Schedule:")
    print(f"    Phase 1: {INFERENCE_DURATION_HOURS}h inference at 320x320")
    print(f"    Phase 2: {COOLDOWN_DURATION_HOURS}h cooldown")
    print(f"    Phase 3: {INFERENCE_DURATION_HOURS}h inference at 640x640")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────
    print("[1/5] Loading ONNX model...")
    if not os.path.isfile(ONNX_MODEL_PATH):
        print(f"  ERROR: '{ONNX_MODEL_PATH}' not found.")
        sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    session = ort.InferenceSession(
        ONNX_MODEL_PATH, sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"  Model loaded. Input: {input_name}")

    # ── Load images into memory ────────────────────────────────────────
    print("\n[2/5] Loading images into memory...")
    if not os.path.isdir(IMAGE_DIR):
        print(f"  ERROR: '{IMAGE_DIR}' not found. Run 02_download_coco_sample.py first.")
        sys.exit(1)

    image_paths = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Pre-load all images into memory to avoid disk I/O during the test
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    print(f"  Loaded {len(images)} images into memory.")
    if len(images) == 0:
        print("  ERROR: No valid images found.")
        sys.exit(1)

    # ── Phase 1: 320x320 ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[3/5] PHASE 1: Inference at 320x320 for {INFERENCE_DURATION_HOURS} hours")
    print(f"{'=' * 60}")

    samples_320, count_320 = run_inference_phase(
        session, input_name, images, 320, INFERENCE_DURATION_HOURS
    )

    result_320 = {
        "image_size": 320,
        "duration_hours": INFERENCE_DURATION_HOURS,
        "total_inferences": count_320,
        "samples": samples_320,
    }
    with open(os.path.join(RESULTS_DIR, "stress_320.json"), "w") as f:
        json.dump(result_320, f, indent=2)
    print(f"\n  Phase 1 complete: {count_320} total inferences.")
    print(f"  Results saved to {RESULTS_DIR}/stress_320.json")

    # ── Phase 2: Cooldown ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[4/5] PHASE 2: Cooldown for {COOLDOWN_DURATION_HOURS} hour(s)")
    print(f"{'=' * 60}")

    cooldown_samples = run_cooldown(COOLDOWN_DURATION_HOURS)

    with open(os.path.join(RESULTS_DIR, "cooldown.json"), "w") as f:
        json.dump({"duration_hours": COOLDOWN_DURATION_HOURS, "samples": cooldown_samples}, f, indent=2)
    print(f"\n  Cooldown complete.")
    print(f"  Results saved to {RESULTS_DIR}/cooldown.json")

    # ── Phase 3: 640x640 ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[5/5] PHASE 3: Inference at 640x640 for {INFERENCE_DURATION_HOURS} hours")
    print(f"{'=' * 60}")

    samples_640, count_640 = run_inference_phase(
        session, input_name, images, 640, INFERENCE_DURATION_HOURS
    )

    result_640 = {
        "image_size": 640,
        "duration_hours": INFERENCE_DURATION_HOURS,
        "total_inferences": count_640,
        "samples": samples_640,
    }
    with open(os.path.join(RESULTS_DIR, "stress_640.json"), "w") as f:
        json.dump(result_640, f, indent=2)
    print(f"\n  Phase 3 complete: {count_640} total inferences.")
    print(f"  Results saved to {RESULTS_DIR}/stress_640.json")

    # ── Summary ────────────────────────────────────────────────────────
    def extract_stats(samples):
        temps = [s["cpu_temp_c"] for s in samples if s["cpu_temp_c"] is not None]
        rams = [s["ram_used_percent"] for s in samples]
        return {
            "temp_min_c": min(temps) if temps else None,
            "temp_max_c": max(temps) if temps else None,
            "temp_mean_c": round(sum(temps) / len(temps), 1) if temps else None,
            "ram_min_percent": min(rams),
            "ram_max_percent": max(rams),
            "ram_mean_percent": round(sum(rams) / len(rams), 1),
        }

    summary = {
        "test_date": datetime.now().isoformat(),
        "inference_duration_hours": INFERENCE_DURATION_HOURS,
        "cooldown_duration_hours": COOLDOWN_DURATION_HOURS,
        "phase_320": {
            "total_inferences": count_320,
            "num_samples": len(samples_320),
            **extract_stats(samples_320),
        },
        "cooldown": {
            "num_samples": len(cooldown_samples),
            **extract_stats(cooldown_samples),
        },
        "phase_640": {
            "total_inferences": count_640,
            "num_samples": len(samples_640),
            **extract_stats(samples_640),
        },
    }

    with open(os.path.join(RESULTS_DIR, "stress_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  STRESS TEST COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n  Results saved to ./{RESULTS_DIR}/:")
    print(f"    - stress_320.json       ({len(samples_320)} samples, {count_320} inferences)")
    print(f"    - cooldown.json         ({len(cooldown_samples)} samples)")
    print(f"    - stress_640.json       ({len(samples_640)} samples, {count_640} inferences)")
    print(f"    - stress_summary.json   (overall summary)")
    print()
    print(f"  320x320: temp {summary['phase_320']['temp_min_c']}–{summary['phase_320']['temp_max_c']}°C, "
          f"RAM {summary['phase_320']['ram_mean_percent']}% avg")
    print(f"  640x640: temp {summary['phase_640']['temp_min_c']}–{summary['phase_640']['temp_max_c']}°C, "
          f"RAM {summary['phase_640']['ram_mean_percent']}% avg")
    print(f"\n  Plot with: python 05_plot_stress.py")
    print()


if __name__ == "__main__":
    main()
