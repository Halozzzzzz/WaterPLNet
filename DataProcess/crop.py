import argparse
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def check_dir(path):
    os.makedirs(path, exist_ok=True)


def load_image_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_label_gray(path):
    return np.array(Image.open(path).convert("L"))


def build_window_starts(length, size, stride):
    if length < size:
        raise ValueError(f"Image side length {length} is smaller than crop size {size}.")
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def crop_and_split(img, label, mask, save_root, index_dict, size=256, stride=128):
    h, w = img.shape[:2]
    x_list = build_window_starts(w, size, stride)
    y_list = build_window_starts(h, size, stride)

    patches = []
    for y in y_list:
        for x in x_list:
            patches.append((
                img[y:y + size, x:x + size],
                label[y:y + size, x:x + size],
                mask[y:y + size, x:x + size],
            ))

    train, temp = train_test_split(patches, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    subsets = {"train": train, "val": val, "test": test}
    for subset_name, subset_data in subsets.items():
        for subfolder in ["img", "label", "mask"]:
            check_dir(os.path.join(save_root, subset_name, subfolder))

        for out, gt, mask_patch in subset_data:
            index_dict[subset_name] += 1
            filename = f"{index_dict[subset_name]:05d}.png"
            Image.fromarray(out).save(os.path.join(save_root, subset_name, "img", filename))
            Image.fromarray(gt).save(os.path.join(save_root, subset_name, "label", filename))
            Image.fromarray(mask_patch).save(os.path.join(save_root, subset_name, "mask", filename))


def batch_process(image_dir, label_dir, mask_dir, save_root, size=256, stride=128):
    check_dir(save_root)
    valid_ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext))
    index_dict = {"train": 0, "val": 0, "test": 0}

    for fname in tqdm(files, desc="Cropping images"):
        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not (os.path.exists(label_path) and os.path.exists(mask_path)):
            print(f"Skip {fname}: label or mask file is missing.")
            continue

        img = load_image_rgb(img_path)
        label = load_label_gray(label_path)
        mask = load_label_gray(mask_path)
        if img.shape[:2] != label.shape or label.shape != mask.shape:
            raise ValueError(f"{fname}: image, label, and mask sizes do not match.")

        crop_and_split(img, label, mask, save_root, index_dict, size=size, stride=stride)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="directory containing SAR images")
    parser.add_argument("--label_dir", required=True, help="directory containing dense labels")
    parser.add_argument("--mask_dir", required=True, help="directory containing low-backscatter masks")
    parser.add_argument("--save_root", required=True, help="output dataset directory")
    parser.add_argument("--size", type=int, default=256, help="crop size")
    parser.add_argument("--stride", type=int, default=128, help="crop stride")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_process(
        args.image_dir,
        args.label_dir,
        args.mask_dir,
        args.save_root,
        size=args.size,
        stride=args.stride,
    )
