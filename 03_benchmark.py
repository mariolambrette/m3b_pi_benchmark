#!/usr/bin/env python3
"""
03_benchmark.py
Benchmark YOLO11n on Raspberry Pi 4 with COCO sample images.

Measures:
  - Inference time (per-image and total)
  - Memory usage (RSS, peak)
  - CPU utilization
  - CPU temperature (proxy for power/thermal load)

Runs at two resolutions: 320x320 and 640x640
Exports model to NCNN format for optimal Pi performance.

Usage:
    python 03_benchmark.py
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

# ── Configuration ──────────────────────────────────────────────────────────
IMAGE_DIR = "coco_sample"
RESULTS_DIR = "results"
MODEL_NAME = "yolo11n"
IMAGE_SIZES = [320, 640]
EXPORT_FORMAT = "ncnn"      # Most efficient for Raspberry Pi
WARMUP_RUNS = 5             # Warmup inferences before timing


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
    }

    # Raspberry Pi specific info
    try:
        with open("/proc/device-tree/model", "r") as f:
            info["pi_model"] = f.read().strip().rstrip('\x00')
    except FileNotFoundError:
        info["pi_model"] = "Unknown (not a Raspberry Pi or model file missing)"

    # CPU frequency
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
        # Fallback: read directly from sysfs
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


def export_model(model_pt_path, export_format=EXPORT_FORMAT):
    """Export YOLO model to the specified format."""
    from ultralytics import YOLO

    print(f"\n  Loading {model_pt_path}...")
    model = YOLO(model_pt_path)

    print(f"  Exporting to {export_format.upper()} format...")
    export_path = model.export(format=export_format)
    print(f"  Model exported to: {export_path}")
    return export_path


def run_benchmark(model_path, image_dir, imgsz, warmup=WARMUP_RUNS):
    """
    Run inference benchmark at a given image size.

    Returns a dict with timing, memory, and temperature metrics.
    """
    from ultralytics import YOLO

    # Collect image paths (skip manifest.json)
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

    # Load model
    print(f"  Loading model from {model_path}...")
    mem_before_load = get_memory_usage()
    model = YOLO(model_path)
    mem_after_load = get_memory_usage()

    model_load_mem_mb = mem_after_load["rss_mb"] - mem_before_load["rss_mb"]
    print(f"  Model loaded (memory delta: +{model_load_mem_mb:.1f} MB)")

    # ── Warmup ─────────────────────────────────────────────────────────
    print(f"  Running {warmup} warmup inferences...")
    for i in range(warmup):
        model(image_paths[0], imgsz=imgsz, verbose=False)

    # ── Benchmark ──────────────────────────────────────────────────────
    print(f"  Running benchmark on {num_images} images...")
    temp_before = get_cpu_temperature()
    mem_before = get_memory_usage()
    cpu_percent_start = psutil.cpu_percent(interval=None)  # reset counter

    inference_times = []
    detections_per_image = []
    peak_rss_mb = 0

    for i, img_path in enumerate(image_paths):
        t0 = time.perf_counter()
        results = model(img_path, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        inference_times.append(elapsed_ms)

        # Count detections
        num_dets = len(results[0].boxes) if results and results[0].boxes is not None else 0
        detections_per_image.append(num_dets)

        # Track peak memory
        current_rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        peak_rss_mb = max(peak_rss_mb, current_rss)

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            avg_so_far = sum(inference_times) / len(inference_times)
            sys.stdout.write(
                f"\r  Progress: {i+1}/{num_images} | "
                f"Avg: {avg_so_far:.1f}ms | "
                f"Last: {elapsed_ms:.1f}ms"
            )
            sys.stdout.flush()

    print()  # newline after progress

    cpu_percent_avg = psutil.cpu_percent(interval=None)
    temp_after = get_cpu_temperature()
    mem_after = get_memory_usage()

    # ── Compute statistics ─────────────────────────────────────────────
    inference_times_sorted = sorted(inference_times)
    total_time_s = sum(inference_times) / 1000

    results_dict = {
        "image_size": imgsz,
        "num_images": num_images,
        "warmup_runs": warmup,
        "timing": {
            "total_seconds": round(total_time_s, 2),
            "mean_ms": round(sum(inference_times) / len(inference_times), 2),
            "median_ms": round(inference_times_sorted[len(inference_times) // 2], 2),
            "min_ms": round(min(inference_times), 2),
            "max_ms": round(max(inference_times), 2),
            "p95_ms": round(
                inference_times_sorted[int(len(inference_times) * 0.95)], 2
            ),
            "p99_ms": round(
                inference_times_sorted[int(len(inference_times) * 0.99)], 2
            ),
            "fps": round(num_images / total_time_s, 2),
        },
        "memory": {
            "before_inference_rss_mb": mem_before["rss_mb"],
            "after_inference_rss_mb": mem_after["rss_mb"],
            "peak_rss_mb": round(peak_rss_mb, 2),
            "model_load_delta_mb": round(model_load_mem_mb, 2),
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
        "per_image_times_ms": [round(t, 2) for t in inference_times],
    }

    return results_dict


def generate_report(system_info, model_info, benchmark_results):
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# YOLO11n Raspberry Pi 4 Benchmark Report")
    lines.append(f"\n**Generated:** {system_info['timestamp']}")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Pi Model | {system_info.get('pi_model', 'N/A')} |")
    lines.append(f"| Platform | {system_info['platform']} |")
    lines.append(f"| Architecture | {system_info['machine']} |")
    lines.append(f"| CPU Cores | {system_info['cpu_count']} |")
    lines.append(f"| CPU Freq | {system_info.get('cpu_freq_mhz', 'N/A')} MHz |")
    lines.append(f"| Total RAM | {system_info['total_ram_gb']} GB |")
    lines.append(f"| Available RAM | {system_info['available_ram_gb']} GB |")
    lines.append(f"| Python | {system_info['python_version']} |")
    lines.append("")

    # Model info
    lines.append("## Model Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    for k, v in model_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Results per image size
    for result in benchmark_results:
        imgsz = result["image_size"]
        lines.append(f"## Results — {imgsz}x{imgsz}")
        lines.append("")

        # Timing
        t = result["timing"]
        lines.append("### Inference Timing")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total time | {t['total_seconds']}s |")
        lines.append(f"| Mean | {t['mean_ms']} ms |")
        lines.append(f"| Median | {t['median_ms']} ms |")
        lines.append(f"| Min | {t['min_ms']} ms |")
        lines.append(f"| Max | {t['max_ms']} ms |")
        lines.append(f"| P95 | {t['p95_ms']} ms |")
        lines.append(f"| P99 | {t['p99_ms']} ms |")
        lines.append(f"| FPS | {t['fps']} |")
        lines.append("")

        # Memory
        m = result["memory"]
        lines.append("### Memory Usage")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Model load delta | +{m['model_load_delta_mb']} MB |")
        lines.append(f"| Peak process RSS | {m['peak_rss_mb']} MB |")
        lines.append(f"| System RAM before | {m['system_available_before_mb']} MB available |")
        lines.append(f"| System RAM after | {m['system_available_after_mb']} MB available |")
        lines.append(f"| System RAM used | {m['system_used_percent']}% |")
        lines.append("")

        # Thermal / Power
        th = result["thermal"]
        lines.append("### Thermal & CPU")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| CPU temp before | {th['temp_before_c']}°C |")
        lines.append(f"| CPU temp after | {th['temp_after_c']}°C |")
        lines.append(f"| Temp delta | {th['temp_delta_c']}°C |")
        lines.append(f"| Avg CPU utilization | {result['cpu']['avg_utilization_percent']}% |")
        lines.append("")

        # Detections
        d = result["detections"]
        lines.append("### Detection Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total detections | {d['total']} |")
        lines.append(f"| Mean per image | {d['mean_per_image']} |")
        lines.append(f"| Max per image | {d['max_per_image']} |")
        lines.append("")

    # Comparison table
    if len(benchmark_results) > 1:
        lines.append("## Comparison: 320 vs 640")
        lines.append("")
        lines.append("| Metric | 320x320 | 640x640 | Ratio |")
        lines.append("|--------|---------|---------|-------|")

        r320 = benchmark_results[0]
        r640 = benchmark_results[1]

        mean320 = r320["timing"]["mean_ms"]
        mean640 = r640["timing"]["mean_ms"]
        lines.append(
            f"| Mean inference | {mean320} ms | {mean640} ms | "
            f"{round(mean640 / mean320, 2)}x |"
        )

        fps320 = r320["timing"]["fps"]
        fps640 = r640["timing"]["fps"]
        lines.append(f"| FPS | {fps320} | {fps640} | {round(fps320 / fps640, 2)}x |")

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

    # Notes on power measurement
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


def main():
    print("=" * 60)
    print("  YOLO11n Raspberry Pi 4 Benchmark")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── System Info ────────────────────────────────────────────────────
    print("\n[1/5] Collecting system information...")
    system_info = get_system_info()
    print(f"  Platform: {system_info['platform']}")
    print(f"  RAM: {system_info['total_ram_gb']} GB total, "
          f"{system_info['available_ram_gb']} GB available")

    # ── Export Model ───────────────────────────────────────────────────
    print("\n[2/5] Preparing YOLO11n model...")
    pt_model = f"{MODEL_NAME}.pt"

    # Download the model weights (Ultralytics does this automatically)
    from ultralytics import YOLO
    model = YOLO(pt_model)  # downloads if not present

    model_info = {
        "Model": "YOLO11n (nano)",
        "Source format": "PyTorch (.pt)",
        "Export format": EXPORT_FORMAT.upper(),
        "Parameters": "~2.6M",
        "Task": "Object Detection",
    }

    # Export to NCNN
    ncnn_model_path = export_model(pt_model, EXPORT_FORMAT)
    model_info["Exported path"] = str(ncnn_model_path)

    # Free memory from PyTorch model
    del model

    # ── Check images exist ─────────────────────────────────────────────
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

        result = run_benchmark(ncnn_model_path, IMAGE_DIR, imgsz)
        benchmark_results.append(result)

        # Save individual result
        result_file = os.path.join(RESULTS_DIR, f"benchmark_{imgsz}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {result_file}")

    # ── Generate Report ────────────────────────────────────────────────
    print(f"\n[5/5] Generating report...")

    # Save combined JSON
    combined = {
        "system_info": system_info,
        "model_info": model_info,
        "benchmarks": benchmark_results,
    }
    combined_file = os.path.join(RESULTS_DIR, "benchmark_combined.json")
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)

    # Save Markdown report
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
    print(f"    - benchmark_combined.json (full data)")
    print(f"    - benchmark_320.json")
    print(f"    - benchmark_640.json")

    # Quick summary
    for r in benchmark_results:
        sz = r["image_size"]
        print(f"\n  [{sz}x{sz}] Mean: {r['timing']['mean_ms']}ms | "
              f"FPS: {r['timing']['fps']} | "
              f"Peak RAM: {r['memory']['peak_rss_mb']}MB")

    print()


if __name__ == "__main__":
    main()
