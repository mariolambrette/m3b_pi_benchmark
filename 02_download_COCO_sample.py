#!/usr/bin/env python3
"""
02_download_coco_sample.py
Download 100 random images from the COCO 2017 validation set.

Uses the COCO API to fetch image metadata, then downloads images directly.
This avoids downloading the entire 6GB+ validation set.

Usage:
    python 02_download_coco_sample.py
"""

import json
import os
import random
import urllib.request
import sys

# ── Configuration ──────────────────────────────────────────────────────────
NUM_IMAGES = 100
SEED = 42
OUTPUT_DIR = "coco_sample"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_ANNOTATION = "annotations/instances_val2017.json"
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"


def download_annotations():
    """Download and extract COCO val2017 annotations if not present."""
    annotation_file = COCO_VAL_ANNOTATION
    if os.path.exists(annotation_file):
        print(f"  Annotations already exist at {annotation_file}")
        return annotation_file

    zip_path = "annotations_trainval2017.zip"
    if not os.path.exists(zip_path):
        print("  Downloading COCO annotations (252MB)... This may take a while.")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        print("  Download complete.")

    print("  Extracting annotations...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract(COCO_VAL_ANNOTATION)
    print("  Extraction complete.")
    return annotation_file


def select_random_images(annotation_file, n=NUM_IMAGES, seed=SEED):
    """Select n random image entries from COCO annotations."""
    print(f"  Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    print(f"  Total images in val2017: {len(images)}")

    random.seed(seed)
    selected = random.sample(images, min(n, len(images)))
    print(f"  Selected {len(selected)} random images (seed={seed})")
    return selected


def download_images(image_list, output_dir=OUTPUT_DIR):
    """Download selected images to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save the image manifest for reproducibility
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(image_list, f, indent=2)

    print(f"  Downloading {len(image_list)} images to {output_dir}/...")
    downloaded = 0
    for i, img in enumerate(image_list):
        filename = img['file_name']
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            downloaded += 1
            continue

        url = f"{COCO_IMAGE_BASE_URL}/{filename}"
        try:
            urllib.request.urlretrieve(url, filepath)
            downloaded += 1
            # Progress indicator
            sys.stdout.write(f"\r  Downloaded {downloaded}/{len(image_list)}")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  WARNING: Failed to download {filename}: {e}")

    print(f"\n  Successfully downloaded {downloaded}/{len(image_list)} images.")
    print(f"  Manifest saved to {manifest_path}")
    return output_dir


def main():
    print("=" * 50)
    print("  COCO Sample Image Downloader")
    print("=" * 50)

    print("\n[1/3] Fetching COCO annotations...")
    annotation_file = download_annotations()

    print("\n[2/3] Selecting random images...")
    selected_images = select_random_images(annotation_file)

    print("\n[3/3] Downloading images...")
    output_dir = download_images(selected_images)

    print(f"\n{'=' * 50}")
    print(f"  Done! {NUM_IMAGES} images saved to ./{output_dir}/")
    print(f"  Next: python 03_benchmark.py")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
