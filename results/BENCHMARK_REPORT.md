# YOLO11n Raspberry Pi 4 Benchmark Report

**Generated:** 2026-02-27T13:43:18.163300

## System Information

| Property | Value |
|----------|-------|
| Pi Model | Raspberry Pi 4 Model B Rev 1.2 |
| Platform | Linux-6.12.47+rpt-rpi-v8-aarch64-with-glibc2.41 |
| Architecture | aarch64 |
| CPU Cores | 4 |
| CPU Freq | 1500.0 MHz |
| Total RAM | 3.71 GB |
| Available RAM | 2.92 GB |
| Python | 3.14.3 |
| ONNX Runtime | 1.24.2 |
| ORT Providers | AzureExecutionProvider, CPUExecutionProvider |

## Model Information

| Property | Value |
|----------|-------|
| Model | YOLO11n (nano) |
| Format | ONNX |
| Runtime | ONNX Runtime 1.24.2 |
| Provider | CPUExecutionProvider |
| Parameters | ~2.6M |
| Task | Object Detection |
| Input shape | ['batch', 3, 'height', 'width'] |
| Output shape | ['batch', 84, 'anchors'] |
| Model load memory | +24.5 MB |
| Threads | 4 |

## Results — 320x320

### Inference Timing (model only)

| Metric | Value |
|--------|-------|
| Total time | 13.21s |
| Mean | 132.1 ms |
| Median | 127.69 ms |
| Min | 118.52 ms |
| Max | 192.85 ms |
| P95 | 156.17 ms |
| P99 | 192.85 ms |
| FPS | 7.57 |

### Full Pipeline Timing (preprocess + inference + postprocess)

| Stage | Mean (ms) |
|-------|-----------|
| Preprocess | 4.1 |
| Inference | 132.1 |
| Postprocess | 2.28 |
| **Total** | **138.47** |
| Pipeline FPS | 7.22 |

### Memory Usage

| Metric | Value |
|--------|-------|
| Peak process RSS | 117.06 MB |
| System RAM before | 2950.29 MB available |
| System RAM after | 2949.2 MB available |
| System RAM used | 22.3% |

### Thermal & CPU

| Metric | Value |
|--------|-------|
| CPU temp before | 43.329°C |
| CPU temp after | 46.251°C |
| Temp delta | 2.9°C |
| Avg CPU utilization | 99.9% |

### Detection Summary

| Metric | Value |
|--------|-------|
| Total detections | 307 |
| Mean per image | 3.07 |
| Max per image | 12 |

## Results — 640x640

### Inference Timing (model only)

| Metric | Value |
|--------|-------|
| Total time | 51.3s |
| Mean | 512.97 ms |
| Median | 501.25 ms |
| Min | 470.83 ms |
| Max | 830.07 ms |
| P95 | 650.18 ms |
| P99 | 830.07 ms |
| FPS | 1.95 |

### Full Pipeline Timing (preprocess + inference + postprocess)

| Stage | Mean (ms) |
|-------|-----------|
| Preprocess | 8.83 |
| Inference | 512.97 |
| Postprocess | 6.81 |
| **Total** | **528.61** |
| Pipeline FPS | 1.89 |

### Memory Usage

| Metric | Value |
|--------|-------|
| Peak process RSS | 157.38 MB |
| System RAM before | 2902.54 MB available |
| System RAM after | 2908.5 MB available |
| System RAM used | 23.4% |

### Thermal & CPU

| Metric | Value |
|--------|-------|
| CPU temp before | 46.251°C |
| CPU temp after | 49.173°C |
| Temp delta | 2.9°C |
| Avg CPU utilization | 98.9% |

### Detection Summary

| Metric | Value |
|--------|-------|
| Total detections | 464 |
| Mean per image | 4.64 |
| Max per image | 26 |

## Comparison: 320 vs 640

| Metric | 320x320 | 640x640 | Ratio |
|--------|---------|---------|-------|
| Mean inference | 132.1 ms | 512.97 ms | 3.88x |
| Pipeline FPS | 7.22 | 1.89 | 3.82x |
| Peak RSS | 117.06 MB | 157.38 MB | 1.34x |
| Total detections | 307 | 464 | — |

## Notes on Power Measurement

True power consumption cannot be measured via software alone on the Raspberry Pi. The CPU temperature delta serves as a rough proxy for thermal/power load. For accurate power measurement, use an external USB power meter (e.g., a USB-C inline power monitor) between your power supply and the Pi.

Typical Pi 4 power draw: ~3W idle, ~6-7W under full CPU load. YOLO inference will push closer to the upper end.
