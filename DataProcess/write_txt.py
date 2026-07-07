import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_txt(data_root, split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(data_root, split, "img")
    out_path = os.path.join(out_dir, f"{split}.txt")

    if not os.path.exists(img_dir):
        print(f"Skip {split}: image directory not found: {img_dir}")
        return

    filenames = sorted(os.listdir(img_dir))
    with open(out_path, "w", encoding="utf-8") as file:
        for fname in filenames:
            file.write(f"{fname}\n")

    print(f"Wrote {out_path} ({len(filenames)} samples)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "dataset" / "MSLCC"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "datafiles" / "MSLCC"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for split in args.splits:
        write_txt(args.data_root, split, args.out_dir)
