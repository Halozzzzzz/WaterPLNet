import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calculate_mean_std(img_dir):
    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    file_list = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext))
    if not file_list:
        raise ValueError(f"No image files found in {img_dir}")

    sum_means = np.zeros(3, dtype=np.float64)
    sum_stds = np.zeros(3, dtype=np.float64)

    for filename in tqdm(file_list, desc="Calculating mean/std"):
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32).reshape(-1, 3)
        sum_means += np.mean(img_np, axis=0)
        sum_stds += np.std(img_np, axis=0)

    mean_result = sum_means / len(file_list)
    std_result = sum_stds / len(file_list)
    return mean_result, std_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=str(PROJECT_ROOT / "dataset" / "MSLCC" / "train" / "img"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mean, std = calculate_mean_std(args.img_dir)
    print("Dataset Mean (per channel, 0-255):", mean)
    print("Dataset Std  (per channel, 0-255):", std)
