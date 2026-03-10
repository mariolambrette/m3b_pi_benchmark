#!/usr/bin/env python3
"""
05_plot_stress.py
Plot the results from 04_stress_test.py.

Generates a single PNG with subplots showing CPU temperature, RAM usage,
and inference throughput across all three phases (320, cooldown, 640).

Usage:
    python 05_plot_stress.py

Requires:
    pip install matplotlib

Output:
    results/stress_test/stress_test_plot.png
"""

import json
import os
import sys

RESULTS_DIR = os.path.join("results", "stress_test")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. Run 04_stress_test.py first.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for Pi
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not found. Install it:")
        print("  pip install matplotlib")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    data_320 = load_json("stress_320.json")
    data_cool = load_json("cooldown.json")
    data_640 = load_json("stress_640.json")

    # Build continuous timeline (minutes from start)
    phase1_duration = data_320["duration_hours"] * 60
    cooldown_duration = data_cool["duration_hours"] * 60
    phase2_offset = phase1_duration
    phase3_offset = phase1_duration + cooldown_duration

    # Extract time series
    t_320 = [s["elapsed_minutes"] for s in data_320["samples"]]
    temp_320 = [s["cpu_temp_c"] for s in data_320["samples"]]
    ram_320 = [s["ram_used_percent"] for s in data_320["samples"]]
    inf_320 = [s.get("inferences_since_last", 0) for s in data_320["samples"]]

    t_cool = [s["elapsed_minutes"] + phase2_offset for s in data_cool["samples"]]
    temp_cool = [s["cpu_temp_c"] for s in data_cool["samples"]]
    ram_cool = [s["ram_used_percent"] for s in data_cool["samples"]]

    t_640 = [s["elapsed_minutes"] + phase3_offset for s in data_640["samples"]]
    temp_640 = [s["cpu_temp_c"] for s in data_640["samples"]]
    ram_640 = [s["ram_used_percent"] for s in data_640["samples"]]
    inf_640 = [s.get("inferences_since_last", 0) for s in data_640["samples"]]

    # Combined timelines
    t_all = t_320 + t_cool + t_640
    temp_all = temp_320 + temp_cool + temp_640
    ram_all = ram_320 + ram_cool + ram_640

    # ── Create plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("YOLO11n Raspberry Pi 4 — Stress Test Results", fontsize=14, fontweight="bold")

    # Phase shading helper
    def shade_phases(ax):
        ax.axvspan(0, phase1_duration, alpha=0.08, color="blue", label="_320 phase")
        ax.axvspan(phase2_offset, phase3_offset, alpha=0.08, color="gray", label="_cooldown")
        ax.axvspan(phase3_offset, t_all[-1] if t_all else phase3_offset + 180,
                   alpha=0.08, color="red", label="_640 phase")

    # --- Plot 1: CPU Temperature ---
    ax1 = axes[0]
    shade_phases(ax1)
    ax1.plot(t_320, temp_320, color="tab:blue", linewidth=1, label="320×320")
    ax1.plot(t_cool, temp_cool, color="gray", linewidth=1, linestyle="--", label="Cooldown")
    ax1.plot(t_640, temp_640, color="tab:red", linewidth=1, label="640×640")
    ax1.set_ylabel("CPU Temperature (°C)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Add throttle warning line at 80°C
    if any(t for t in temp_all if t and t > 70):
        ax1.axhline(y=80, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
        ax1.annotate("Throttle threshold (80°C)", xy=(t_all[-1] * 0.02, 81),
                     fontsize=8, color="red", alpha=0.7)

    # --- Plot 2: RAM Usage ---
    ax2 = axes[1]
    shade_phases(ax2)
    ax2.plot(t_320, ram_320, color="tab:blue", linewidth=1, label="320×320")
    ax2.plot(t_cool, ram_cool, color="gray", linewidth=1, linestyle="--", label="Cooldown")
    ax2.plot(t_640, ram_640, color="tab:red", linewidth=1, label="640×640")
    ax2.set_ylabel("RAM Used (%)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # --- Plot 3: Inference Throughput ---
    ax3 = axes[2]
    shade_phases(ax3)
    # Skip first sample (no inferences yet)
    ax3.plot(t_320[1:], inf_320[1:], color="tab:blue", linewidth=1, label="320×320")
    ax3.plot(t_640[1:], inf_640[1:], color="tab:red", linewidth=1, label="640×640")
    ax3.set_ylabel("Inferences / minute")
    ax3.set_xlabel("Time (minutes)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Phase labels
    for ax in axes:
        y_top = ax.get_ylim()[1]
        ax.text(phase1_duration / 2, y_top * 0.95, "320×320",
                ha="center", fontsize=9, alpha=0.4, fontweight="bold")
        ax.text(phase2_offset + cooldown_duration / 2, y_top * 0.95, "Cooldown",
                ha="center", fontsize=9, alpha=0.4, fontweight="bold")
        ax.text(phase3_offset + phase1_duration / 2, y_top * 0.95, "640×640",
                ha="center", fontsize=9, alpha=0.4, fontweight="bold")

    # Format x-axis as hours
    ax3.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/60:.1f}h")
    )
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(60))

    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "stress_test_plot.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # ── Also print a text summary ──────────────────────────────────────
    summary = load_json("stress_summary.json")
    print("\nSummary:")
    for phase_key, label in [("phase_320", "320×320"), ("phase_640", "640×640")]:
        p = summary[phase_key]
        print(f"  {label}: {p['total_inferences']} inferences | "
              f"Temp: {p['temp_min_c']}–{p['temp_max_c']}°C (mean {p['temp_mean_c']}°C) | "
              f"RAM: {p['ram_mean_percent']}% avg")
    cool = summary["cooldown"]
    print(f"  Cooldown: Temp: {cool['temp_min_c']}–{cool['temp_max_c']}°C | "
          f"RAM: {cool['ram_mean_percent']}% avg")


if __name__ == "__main__":
    main()
