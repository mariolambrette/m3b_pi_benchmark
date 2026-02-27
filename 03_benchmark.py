#!/usr/bin/env python3
"""
03_benchmark.py
Benchmark YOLO11n on Raspberry Pi 4 with COCO sample images.

Uses ONNX Runtime directly — no PyTorch or ultralytics needed on the Pi.

Measures:
  - Inference time (per-image and total)
  - Memory usage (RSS, peak)
  - CPU utilization
  - CPU temperature (proxy for power/thermal load)

Runs at two resolutions: 320x320 and 640x640.

Usage:
    python 03_benchmark.py
"""

import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import psutil

# ── Configuration ──────────────────────────────────────────────────────────
IMAGE_DIR = "coco_sample"
RESULTS_DIR = "results"
ONNX_MODEL_PATH = "yolo11n.onnx"
IMAGE_SIZES = [320, 640]
WARMUP_RUNS = 5
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_CLASSES = 80  # COCO


# ── YOLO Pre/Post Processing ──────────────────────────────────────────────

def letterbox(img, new_shape=(640, 640)):
    """
    Resize and pad image to target size, preserving aspect ratio.
    Returns the padded image, scale ratio, and padding offsets.
    """
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
    return img, r, (dw, dh)


def preprocess(img_bgr, imgsz):
    """
    Preprocess a BGR image for YOLO ONNX inference.
    Returns a float32 NCHW tensor normalized to [0, 1].
    """
    img, ratio, pad = letterbox(img_bgr, new_shape=(imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img, ratio, pad


def postprocess(output, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """
    Post-process YOLO ONNX output.

    YOLO11 ONNX output shape: [1, 84, N]
      - 84 = 4 (cx, cy, w, h) + 80 (class scores)
      - N = number of candidate detections

    Returns list of (x1, y1, x2, y2, confidence, class_id) tuples.
    """
    # Transpose to [N, 84]
    preds = output[0].squeeze(0).T  # [N, 84]

    # Split box coords and class scores
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    # Get best class per detection
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence
    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert cx,cy,w,h -> x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

    # NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(
        boxes_cxcywh[:, :4].tolist(),  # cv2 NMS wants [cx, cy, w, h]
        max_scores.tolist(),
        conf_thresh,
        iou_thresh,
    )

    detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detections.append((
                float(boxes_xyxy[i, 0]),
                float(boxes_xyxy[i, 1]),
                float(boxes_xyxy[i, 2]),
                float(boxes_xyxy[i, 3]),
                float(max_scores[i]),
                int(class_ids[i]),
            ))

    return detections


# ── System Monitoring ──────────────────────────────────────────────────────

def get_system_info():
    """Collect system information for the report."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "onnxruntime_version": ort.__version__,
        "ort_providers": ort.get_available_providers(),
    }

    # Raspberry Pi specific info
    try:
        with open("/proc/device-tree/model", "r") as f:
            info["pi_model"] = f.read().strip().rstrip('\x00')
    except FileNotFoundError:
        info["pi_model"] = "Unknown"

    try:
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz"] = freq.current
    except Exception:
        pass

    return info


def get_cpu_temperature():
    """Read CPU temperature on Raspberry Pi."""
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            return temps["cpu_thermal"][0].current
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": round(mem_info.rss / (1024**2), 2),
        "vms_mb": round(mem_info.vms / (1024**2), 2),
        "system_available_mb": round(
            psutil.virtual_memory().available / (1024**2), 2
        ),
        "system_used_percent": psutil.virtual_memory().percent,
    }


# ── Benchmark ──────────────────────────────────────────────────────────────

def run_benchmark(session, image_dir, imgsz, input_name, warmup=WARMUP_RUNS):
    """
    Run inference benchmark at a given image size.
    Returns a dict with timing, memory, and temperature metrics.
    """
    # Collect image paths
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_paths:
        print(f"  ERROR: No images found in {image_dir}")
        sys.exit(1)

    num_images = len(image_paths)
    print(f"\n  Found {num_images} images")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Warmup runs: {warmup}")

    # ── Warmup ─────────────────────────────────────────────────────────
    print(f"  Running {warmup} warmup inferences...")
    warmup_img = cv2.imread(image_paths[0])
    for i in range(warmup):
        blob, _, _ = preprocess(warmup_img, imgsz)
        session.run(None, {input_name: blob})

    # ── Benchmark ──────────────────────────────────────────────────────
    print(f"  Running benchmark on {num_images} images...")
    temp_before = get_cpu_temperature()
    mem_before = get_memory_usage()
    cpu_percent_start = psutil.cpu_percent(interval=None)

    inference_times = []
    preprocess_times = []
    postprocess_times = []
    detections_per_image = []
    peak_rss_mb = 0

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"\n  WARNING: Could not read {img_path}, skipping")
            continue

        # Preprocess
        t_pre0 = time.perf_counter()
        blob, ratio, pad = preprocess(img_bgr, imgsz)
        t_pre1 = time.perf_counter()
        preprocess_times.append((t_pre1 - t_pre0) * 1000)

        # Inference
        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        inference_times.append(elapsed_ms)

        # Postprocess
        t_post0 = time.perf_counter()
        dets = postprocess(output)
        t_post1 = time.perf_counter()
        postprocess_times.append((t_post1 - t_post0) * 1000)

        detections_per_image.append(len(dets))

        # Track peak memory
        current_rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        peak_rss_mb = max(peak_rss_mb, current_rss)

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            avg_so_far = sum(inference_times) / len(inference_times)
            sys.stdout.write(
                f"\r  Progress: {i+1}/{num_images} | "
                f"Avg inference: {avg_so_far:.1f}ms | "
                f"Last: {elapsed_ms:.1f}ms"
            )
            sys.stdout.flush()

    print()

    cpu_percent_avg = psutil.cpu_percent(interval=None)
    temp_after = get_cpu_temperature()
    mem_after = get_memory_usage()

    # ── Compute statistics ─────────────────────────────────────────────
    def percentile(sorted_list, p):
        idx = int(len(sorted_list) * p)
        return sorted_list[min(idx, len(sorted_list) - 1)]

    inference_sorted = sorted(inference_times)
    total_inference_s = sum(inference_times) / 1000
    total_pipeline_ms = [
        pre + inf + post
        for pre, inf, post in zip(preprocess_times, inference_times, postprocess_times)
    ]
    total_pipeline_s = sum(total_pipeline_ms) / 1000

    results_dict = {
        "image_size": imgsz,
        "num_images": len(inference_times),
        "warmup_runs": warmup,
        "timing_inference_only": {
            "total_seconds": round(total_inference_s, 2),
            "mean_ms": round(sum(inference_times) / len(inference_times), 2),
            "median_ms": round(percentile(inference_sorted, 0.5), 2),
            "min_ms": round(min(inference_times), 2),
            "max_ms": round(max(inference_times), 2),
            "p95_ms": round(percentile(inference_sorted, 0.95), 2),
            "p99_ms": round(percentile(inference_sorted, 0.99), 2),
            "fps": round(len(inference_times) / total_inference_s, 2),
        },
        "timing_full_pipeline": {
            "mean_preprocess_ms": round(sum(preprocess_times) / len(preprocess_times), 2),
            "mean_inference_ms": round(sum(inference_times) / len(inference_times), 2),
            "mean_postprocess_ms": round(sum(postprocess_times) / len(postprocess_times), 2),
            "mean_total_ms": round(sum(total_pipeline_ms) / len(total_pipeline_ms), 2),
            "total_seconds": round(total_pipeline_s, 2),
            "fps": round(len(total_pipeline_ms) / total_pipeline_s, 2),
        },
        "memory": {
            "before_inference_rss_mb": mem_before["rss_mb"],
            "after_inference_rss_mb": mem_after["rss_mb"],
            "peak_rss_mb": round(peak_rss_mb, 2),
            "system_available_before_mb": mem_before["system_available_mb"],
            "system_available_after_mb": mem_after["system_available_mb"],
            "system_used_percent": mem_after["system_used_percent"],
        },
        "thermal": {
            "temp_before_c": temp_before,
            "temp_after_c": temp_after,
            "temp_delta_c": (
                round(temp_after - temp_before, 1)
                if temp_before and temp_after
                else None
            ),
        },
        "cpu": {
            "avg_utilization_percent": cpu_percent_avg,
        },
        "detections": {
            "total": sum(detections_per_image),
            "mean_per_image": round(
                sum(detections_per_image) / len(detections_per_image), 2
            ),
            "max_per_image": max(detections_per_image),
        },
        "per_image_inference_ms": [round(t, 2) for t in inference_times],
    }

    return results_dict


# ── Report Generation ──────────────────────────────────────────────────────

def generate_report(system_info, model_info, benchmark_results):
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# YOLO11n Raspberry Pi 4 Benchmark Report")
    lines.append(f"\n**Generated:** {system_info['timestamp']}")
    lines.append("")

    lines.append("## System Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Pi Model | {system_info.get('pi_model', 'N/A')} |")
    lines.append(f"| Platform | {system_info['platform']} |")
    lines.append(f"| Architecture | {system_info['machine']} |")
    lines.append(f"| CPU Cores | {system_info['cpu_count']} |")
    lines.append(f"| CPU Freq | {system_info.get('cpu_freq_mhz', 'N/A')} MHz |")
    lines.append(f"| Total RAM | {system_info['total_ram_gb']} GB |")
    lines.append(f"| Available RAM | {system_info['available_ram_gb']} GB |")
    lines.append(f"| Python | {system_info['python_version']} |")
    lines.append(f"| ONNX Runtime | {system_info['onnxruntime_version']} |")
    lines.append(f"| ORT Providers | {', '.join(system_info['ort_providers'])} |")
    lines.append("")

    lines.append("## Model Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    for k, v in model_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    for result in benchmark_results:
        imgsz = result["image_size"]
        lines.append(f"## Results — {imgsz}x{imgsz}")
        lines.append("")

        # Inference-only timing
        t = result["timing_inference_only"]
        lines.append("### Inference Timing (model only)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total time | {t['total_seconds']}s |")
        lines.append(f"| Mean | {t['mean_ms']} ms |")
        lines.append(f"| Median | {t['median_ms']} ms |")
        lines.append(f"| Min | {t['min_ms']} ms |")
        lines.append(f"| Max | {t['max_ms']} ms |")
        lines.append(f"| P95 | {t['p95_ms']} ms |")
        lines.append(f"| P99 | {t['p99_ms']} ms |")
        lines.append(f"| FPS | {t['fps']} |")
        lines.append("")

        # Full pipeline timing
        p = result["timing_full_pipeline"]
        lines.append("### Full Pipeline Timing (preprocess + inference + postprocess)")
        lines.append("")
        lines.append("| Stage | Mean (ms) |")
        lines.append("|-------|-----------|")
        lines.append(f"| Preprocess | {p['mean_preprocess_ms']} |")
        lines.append(f"| Inference | {p['mean_inference_ms']} |")
        lines.append(f"| Postprocess | {p['mean_postprocess_ms']} |")
        lines.append(f"| **Total** | **{p['mean_total_ms']}** |")
        lines.append(f"| Pipeline FPS | {p['fps']} |")
        lines.append("")

        # Memory
        m = result["memory"]
        lines.append("### Memory Usage")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Peak process RSS | {m['peak_rss_mb']} MB |")
        lines.append(f"| System RAM before | {m['system_available_before_mb']} MB available |")
        lines.append(f"| System RAM after | {m['system_available_after_mb']} MB available |")
        lines.append(f"| System RAM used | {m['system_used_percent']}% |")
        lines.append("")

        # Thermal
        th = result["thermal"]
        lines.append("### Thermal & CPU")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| CPU temp before | {th['temp_before_c']}°C |")
        lines.append(f"| CPU temp after | {th['temp_after_c']}°C |")
        lines.append(f"| Temp delta | {th['temp_delta_c']}°C |")
        lines.append(f"| Avg CPU utilization | {result['cpu']['avg_utilization_percent']}% |")
        lines.append("")

        # Detections
        d = result["detections"]
        lines.append("### Detection Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total detections | {d['total']} |")
        lines.append(f"| Mean per image | {d['mean_per_image']} |")
        lines.append(f"| Max per image | {d['max_per_image']} |")
        lines.append("")

    # Comparison
    if len(benchmark_results) > 1:
        lines.append("## Comparison: 320 vs 640")
        lines.append("")
        lines.append("| Metric | 320x320 | 640x640 | Ratio |")
        lines.append("|--------|---------|---------|-------|")

        r320 = benchmark_results[0]
        r640 = benchmark_results[1]

        mean320 = r320["timing_inference_only"]["mean_ms"]
        mean640 = r640["timing_inference_only"]["mean_ms"]
        lines.append(
            f"| Mean inference | {mean320} ms | {mean640} ms | "
            f"{round(mean640 / mean320, 2)}x |"
        )

        fps320 = r320["timing_full_pipeline"]["fps"]
        fps640 = r640["timing_full_pipeline"]["fps"]
        lines.append(f"| Pipeline FPS | {fps320} | {fps640} | {round(fps320 / fps640, 2)}x |")

        peak320 = r320["memory"]["peak_rss_mb"]
        peak640 = r640["memory"]["peak_rss_mb"]
        lines.append(
            f"| Peak RSS | {peak320} MB | {peak640} MB | "
            f"{round(peak640 / peak320, 2)}x |"
        )

        det320 = r320["detections"]["total"]
        det640 = r640["detections"]["total"]
        lines.append(f"| Total detections | {det320} | {det640} | — |")
        lines.append("")

    # Power notes
    lines.append("## Notes on Power Measurement")
    lines.append("")
    lines.append(
        "True power consumption cannot be measured via software alone on the "
        "Raspberry Pi. The CPU temperature delta serves as a rough proxy for "
        "thermal/power load. For accurate power measurement, use an external "
        "USB power meter (e.g., a USB-C inline power monitor) between your "
        "power supply and the Pi."
    )
    lines.append("")
    lines.append(
        "Typical Pi 4 power draw: ~3W idle, ~6-7W under full CPU load. "
        "YOLO inference will push closer to the upper end."
    )
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  YOLO11n Raspberry Pi 4 Benchmark")
    print("  (ONNX Runtime — no PyTorch required)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── System Info ────────────────────────────────────────────────────
    print("\n[1/5] Collecting system information...")
    system_info = get_system_info()
    print(f"  Platform: {system_info['platform']}")
    print(f"  RAM: {system_info['total_ram_gb']} GB total, "
          f"{system_info['available_ram_gb']} GB available")
    print(f"  ONNX Runtime: {system_info['onnxruntime_version']}")
    print(f"  Providers: {system_info['ort_providers']}")

    # ── Load Model ─────────────────────────────────────────────────────
    print("\n[2/5] Loading ONNX model...")
    if not os.path.isfile(ONNX_MODEL_PATH):
        print(f"\n  ERROR: '{ONNX_MODEL_PATH}' not found.")
        print(f"  Export on another machine:")
        print(f"    pip install ultralytics")
        print(f"    yolo export model=yolo11n.pt format=onnx")
        print(f"  Then copy here:")
        print(f"    scp yolo11n.onnx pi@<pi-ip>:~/yolo_benchmark/")
        sys.exit(1)

    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    mem_before_load = get_memory_usage()
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options,
                                   providers=["CPUExecutionProvider"])
    mem_after_load = get_memory_usage()

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    model_load_mem = mem_after_load["rss_mb"] - mem_before_load["rss_mb"]
    print(f"  Model loaded successfully (memory delta: +{model_load_mem:.1f} MB)")
    print(f"  Input:  {input_name} {input_shape}")
    print(f"  Output: {output_shape}")

    model_info = {
        "Model": "YOLO11n (nano)",
        "Format": "ONNX",
        "Runtime": f"ONNX Runtime {ort.__version__}",
        "Provider": "CPUExecutionProvider",
        "Parameters": "~2.6M",
        "Task": "Object Detection",
        "Input shape": str(input_shape),
        "Output shape": str(output_shape),
        "Model load memory": f"+{model_load_mem:.1f} MB",
        "Threads": sess_options.intra_op_num_threads,
    }

    # ── Check images ───────────────────────────────────────────────────
    if not os.path.isdir(IMAGE_DIR):
        print(f"\n  ERROR: Image directory '{IMAGE_DIR}' not found.")
        print(f"  Please run 02_download_coco_sample.py first.")
        sys.exit(1)

    # ── Run benchmarks ─────────────────────────────────────────────────
    benchmark_results = []

    for i, imgsz in enumerate(IMAGE_SIZES):
        step = i + 3
        print(f"\n[{step}/5] Benchmarking at {imgsz}x{imgsz}...")
        print("-" * 50)

        result = run_benchmark(session, IMAGE_DIR, imgsz, input_name)
        benchmark_results.append(result)

        result_file = os.path.join(RESULTS_DIR, f"benchmark_{imgsz}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {result_file}")

    # ── Generate Report ────────────────────────────────────────────────
    print(f"\n[5/5] Generating report...")

    combined = {
        "system_info": system_info,
        "model_info": model_info,
        "benchmarks": benchmark_results,
    }
    combined_file = os.path.join(RESULTS_DIR, "benchmark_combined.json")
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)

    report = generate_report(system_info, model_info, benchmark_results)
    report_file = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md")
    with open(report_file, 'w') as f:
        f.write(report)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n  Results saved to ./{RESULTS_DIR}/:")
    print(f"    - BENCHMARK_REPORT.md  (human-readable report)")
    print(f"    - benchmark_combined.json")
    print(f"    - benchmark_320.json")
    print(f"    - benchmark_640.json")

    for r in benchmark_results:
        sz = r["image_size"]
        t = r["timing_inference_only"]
        p = r["timing_full_pipeline"]
        print(f"\n  [{sz}x{sz}] Inference: {t['mean_ms']}ms | "
              f"Pipeline: {p['mean_total_ms']}ms | "
              f"FPS: {p['fps']} | "
              f"Peak RAM: {r['memory']['peak_rss_mb']}MB")

    print()


if __name__ == "__main__":
    main()
