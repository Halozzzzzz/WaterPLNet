import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def draw_point_from_label(label, kernal_size=100, point_size=3):
    """
    输入:  label (0/1/255)
           0 = 非水体
           1 = 水体
           255 = 未知/背景

    输出: point_mask (0/1/255)
           每个连通区域仅保留一个点
    """
    h, w = label.shape
    point_mask = np.ones((h, w), dtype=np.uint8) * 255  # 默认全背景 255

    # 获取有效类别（排除 255）
    classes = np.unique(label)
    classes = classes[classes != 255]

    for cls in classes:
        # 当前类别的二值 mask
        binary_mask = (label == cls).astype(np.uint8) * 255

        # 找所有连通区域
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < kernal_size:
                continue

            # 找轮廓内部距离边界最远的点
            dist = np.zeros((h, w), dtype=np.float32)
            for i in range(h):
                for j in range(w):
                    dist[i, j] = cv2.pointPolygonTest(contour, (j, i), True)

            _, _, _, max_pt = cv2.minMaxLoc(dist)
            cx, cy = int(max_pt[0]), int(max_pt[1])

            # 将该点赋值为该类别（不是固定的 1，而是 0/1 等类别）
            point_mask[cy:cy+point_size, cx:cx+point_size] = cls

    return point_mask


def draw_random_point_from_label(label, kernal_size=100, point_size=3, rng=None):
    """
    输入:  label (0/1/255)
           0 = 非水体
           1 = 水体
           255 = 未知/背景

    输出: point_mask (0/1/255)
           每个连通区域随机保留一个点
    """
    h, w = label.shape
    point_mask = np.ones((h, w), dtype=np.uint8) * 255
    rng = rng or np.random.default_rng()

    classes = np.unique(label)
    classes = classes[classes != 255]

    for cls in classes:
        binary_mask = (label == cls).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < kernal_size:
                continue

            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], contourIdx=-1, color=1, thickness=-1)
            ys, xs = np.where(region_mask == 1)
            if len(xs) == 0:
                continue

            point_idx = rng.integers(0, len(xs))
            cx, cy = int(xs[point_idx]), int(ys[point_idx])

            y1 = min(cy + point_size, h)
            x1 = min(cx + point_size, w)
            point_mask[cy:y1, cx:x1] = cls

    return point_mask


def draw_noisy_random_point_from_label(label, kernal_size=100, point_size=3,
                                       noise_ratio=0.1, rng=None):
    """
    输入:  label (0/1/255)
           0 = 非水体
           1 = 水体
           255 = 未知/背景

    输出: point_mask (0/1/255)
           每个连通区域随机保留一个点，并按点实例比例翻转部分点标签
    """
    h, w = label.shape
    point_mask = np.ones((h, w), dtype=np.uint8) * 255
    rng = rng or np.random.default_rng()
    noise_ratio = float(np.clip(noise_ratio, 0.0, 1.0))

    classes = np.unique(label)
    classes = classes[classes != 255]

    point_count = 0
    noisy_count = 0
    for cls in classes:
        binary_mask = (label == cls).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < kernal_size:
                continue

            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], contourIdx=-1, color=1, thickness=-1)
            ys, xs = np.where(region_mask == 1)
            if len(xs) == 0:
                continue

            point_idx = rng.integers(0, len(xs))
            cx, cy = int(xs[point_idx]), int(ys[point_idx])

            point_cls = int(cls)
            point_count += 1
            if rng.random() < noise_ratio:
                point_cls = 1 - point_cls
                noisy_count += 1

            y1 = min(cy + point_size, h)
            x1 = min(cx + point_size, w)
            point_mask[cy:y1, cx:x1] = point_cls

    return point_mask, point_count, noisy_count


def flip_existing_point_label(point_mask, noise_ratio=0.1, rng=None):
    """
    对已有点标签中的点实例做随机类别翻转，不重新选择点位置。

    输入/输出约定保持不变：
      0/1: 点标签类别
      255: 未标注区域
    """
    rng = rng or np.random.default_rng()
    noise_ratio = float(np.clip(noise_ratio, 0.0, 1.0))
    noisy_point_mask = point_mask.copy()

    valid = (point_mask != 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(valid, connectivity=8)
    point_count = 0
    noisy_count = 0

    for label_idx in range(1, num_labels):
        region = labels == label_idx
        values = point_mask[region]
        values = values[values != 255]
        if values.size == 0:
            continue

        classes, counts = np.unique(values, return_counts=True)
        point_cls = int(classes[np.argmax(counts)])
        if point_cls not in (0, 1):
            continue

        point_count += 1
        if rng.random() < noise_ratio:
            noisy_point_mask[region] = 1 - point_cls
            noisy_count += 1

    return noisy_point_mask, point_count, noisy_count


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_label_dir(label_dir, save_mask_dir, method='center', seed=None,
                      kernal_size=100, point_size=3, noise_ratio=0.1):
    check_dir(save_mask_dir)
    filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    rng = np.random.default_rng(seed)
    total_points = 0
    total_noisy_points = 0

    for name in tqdm(filenames, desc=f"生成点标签: {os.path.basename(label_dir)}"):
        label_path = os.path.join(label_dir, name)
        label = np.array(Image.open(label_path).convert('L'))

        if method == 'center':
            point_mask = draw_point_from_label(label, kernal_size=kernal_size, point_size=point_size)
        elif method == 'random':
            point_mask = draw_random_point_from_label(
                label,
                kernal_size=kernal_size,
                point_size=point_size,
                rng=rng
            )
        elif method == 'noise':
            point_mask, point_count, noisy_count = draw_noisy_random_point_from_label(
                label,
                kernal_size=kernal_size,
                point_size=point_size,
                noise_ratio=noise_ratio,
                rng=rng
            )
            total_points += point_count
            total_noisy_points += noisy_count
        else:
            raise ValueError(f"Unsupported point label method: {method}")

        Image.fromarray(point_mask).save(os.path.join(save_mask_dir, name))

    return total_points, total_noisy_points


def process_existing_point_dir(point_dir, save_mask_dir, seed=None, noise_ratio=0.1):
    check_dir(save_mask_dir)
    filenames = sorted([f for f in os.listdir(point_dir) if f.endswith('.png')])
    rng = np.random.default_rng(seed)
    total_points = 0
    total_noisy_points = 0

    for name in tqdm(filenames, desc=f"翻转中心点标签: {os.path.basename(point_dir)}"):
        point_path = os.path.join(point_dir, name)
        point_mask = np.array(Image.open(point_path).convert('L'))
        noisy_point_mask, point_count, noisy_count = flip_existing_point_label(
            point_mask,
            noise_ratio=noise_ratio,
            rng=rng
        )
        total_points += point_count
        total_noisy_points += noisy_count
        Image.fromarray(noisy_point_mask).save(os.path.join(save_mask_dir, name))

    return total_points, total_noisy_points


def write_manifest(dataset_root, manifest, output_dir_name):
    manifest_path = os.path.join(dataset_root, f'{output_dir_name}_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def process_dataset(dataset_root, splits, method='random', seed=42,
                    kernal_size=100, point_size=3, noise_ratio=0.1,
                    output_dir_name='point_label_random'):
    manifest = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': method,
        'seed': seed,
        'kernal_size': kernal_size,
        'point_size': point_size,
        'noise_ratio': noise_ratio if method in ('noise', 'center_noise') else None,
        'source_label_dir': 'label',
        'output_point_label_dir': output_dir_name,
        'splits': {}
    }

    for split in splits:
        label_dir = os.path.join(dataset_root, split, 'label')
        save_dir = os.path.join(dataset_root, split, output_dir_name)
        if method == 'center_noise':
            source_point_dir = os.path.join(dataset_root, split, 'point_label')
            if not os.path.isdir(source_point_dir):
                print(f'跳过不存在的目录: {source_point_dir}')
                continue

            total_points, total_noisy_points = process_existing_point_dir(
                source_point_dir,
                save_dir,
                seed=seed,
                noise_ratio=noise_ratio
            )
            manifest['splits'][split] = {
                'label_dir': label_dir,
                'source_point_label_dir': source_point_dir,
                'point_label_dir': save_dir,
                'file_count': len([f for f in os.listdir(save_dir) if f.endswith('.png')]),
                'point_count': total_points,
                'noisy_point_count': total_noisy_points,
                'actual_noise_ratio': (
                    total_noisy_points / total_points if total_points > 0 else None
                )
            }
            continue

        if not os.path.isdir(label_dir):
            print(f'跳过不存在的目录: {label_dir}')
            continue

        total_points, total_noisy_points = process_label_dir(
            label_dir,
            save_dir,
            method=method,
            seed=seed,
            kernal_size=kernal_size,
            point_size=point_size,
            noise_ratio=noise_ratio
        )
        manifest['splits'][split] = {
            'label_dir': label_dir,
            'point_label_dir': save_dir,
            'file_count': len([f for f in os.listdir(save_dir) if f.endswith('.png')]),
            'point_count': total_points if method == 'noise' else None,
            'noisy_point_count': total_noisy_points if method == 'noise' else None,
            'actual_noise_ratio': (
                total_noisy_points / total_points
                if method == 'noise' and total_points > 0 else None
            )
        }

    manifest_path = write_manifest(dataset_root, manifest, output_dir_name)
    print(f'生成记录已保存: {manifest_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=str(PROJECT_ROOT / 'dataset')
    )
    parser.add_argument('--datasets', nargs='+', default=['MSLCC', 'GF3_3m', 'FUSAR'])
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--method', type=str, default='random', choices=['center', 'random', 'noise', 'center_noise'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernal_size', type=int, default=100)
    parser.add_argument('--point_size', type=int, default=3)
    parser.add_argument('--noise_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir_name', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir_name is None:
        if args.method == 'noise':
            args.output_dir_name = 'point_label_noise'
        elif args.method == 'center_noise':
            args.output_dir_name = 'point_label_center_noise'
        else:
            args.output_dir_name = f'point_label_{args.method}'
    for dataset in args.datasets:
        root = os.path.join(args.dataset_root, dataset)
        print(f'开始生成点标签: {root}')
        process_dataset(
            root,
            args.splits,
            method=args.method,
            seed=args.seed,
            kernal_size=args.kernal_size,
            point_size=args.point_size,
            noise_ratio=args.noise_ratio,
            output_dir_name=args.output_dir_name
        )
