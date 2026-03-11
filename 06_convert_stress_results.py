#!/usr/bin/env python3
"""
06_json_to_csv.py
Convert stress test JSON outputs to CSV files for easy use in Excel, R, etc.

Usage:
    python 06_json_to_csv.py

Outputs:
    results/stress_test/stress_320.csv
    results/stress_test/cooldown.csv
    results/stress_test/stress_640.csv
    results/stress_test/stress_combined.csv   (all phases in one file)
"""

import csv
import json
import os
import sys

RESULTS_DIR = os.path.join("results", "stress_test")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        print(f"WARNING: {path} not found, skipping.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def samples_to_csv(samples, output_path, extra_fields=None):
    """Write a list of sample dicts to a CSV file."""
    if not samples:
        return

    # Build fieldnames from first sample + any extras
    fieldnames = list(samples[0].keys())
    if extra_fields:
        fieldnames = list(extra_fields.keys()) + fieldnames

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {**s}
            if extra_fields:
                row.update(extra_fields)
            writer.writerow(row)

    print(f"  {output_path}  ({len(samples)} rows)")


def main():
    print("Converting stress test JSON to CSV...\n")

    combined_samples = []

    # ── 320x320 ────────────────────────────────────────────────────────
    data = load_json("stress_320.json")
    if data:
        samples_to_csv(
            data["samples"],
            os.path.join(RESULTS_DIR, "stress_320.csv"),
        )
        for s in data["samples"]:
            combined_samples.append({"phase": "320x320", **s})

    # ── Cooldown ───────────────────────────────────────────────────────
    data = load_json("cooldown.json")
    if data:
        samples_to_csv(
            data["samples"],
            os.path.join(RESULTS_DIR, "cooldown.csv"),
        )
        for s in data["samples"]:
            combined_samples.append({"phase": "cooldown", **s})

    # ── 640x640 ────────────────────────────────────────────────────────
    data = load_json("stress_640.json")
    if data:
        samples_to_csv(
            data["samples"],
            os.path.join(RESULTS_DIR, "stress_640.csv"),
        )
        for s in data["samples"]:
            combined_samples.append({"phase": "640x640", **s})

    # ── Combined ───────────────────────────────────────────────────────
    if combined_samples:
        combined_path = os.path.join(RESULTS_DIR, "stress_combined.csv")
        fieldnames = list(combined_samples[0].keys())
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_samples)
        print(f"  {combined_path}  ({len(combined_samples)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
