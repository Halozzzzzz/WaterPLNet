import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def filter_connected_regions(mask_dir, point_dir, output_dir):
    check_dir(output_dir)

    for fname in tqdm(os.listdir(mask_dir), desc=f'筛选连通水体：{os.path.basename(mask_dir)}'):
        if not fname.lower().endswith('.png'):
            continue

        mask_path = os.path.join(mask_dir, fname)
        point_path = os.path.join(point_dir, fname)
        save_path = os.path.join(output_dir, fname)

        # 读取图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        point_mask = cv2.imread(point_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or point_mask is None:
            print(f"跳过读取失败的文件: {fname}")
            continue

        # 1. 二值化掩码
        binary_mask = (mask == 1).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)

        # 2. 连通区域分析
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        filtered = np.full_like(binary_mask, 255, dtype=np.uint8)

        matched_labels = np.unique(labels[point_mask == 1])
        matched_labels = matched_labels[matched_labels != 0]
        if len(matched_labels) > 0:
            filtered[np.isin(labels, matched_labels)] = 1

        # 3. 保存结果
        cv2.imwrite(save_path, filtered)


def write_manifest(dataset_root, manifest, output_dir_name):
    manifest_path = os.path.join(dataset_root, f'{output_dir_name}_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def process_dataset(root, subsets, point_dir_name='point_label_random',
                    output_dir_name='domain_random', mask_dir_name='mask'):
    manifest = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_mask_dir': mask_dir_name,
        'source_point_label_dir': point_dir_name,
        'output_domain_dir': output_dir_name,
        'splits': {}
    }

    for subset in subsets:
        mask_dir = os.path.join(root, subset, mask_dir_name)
        point_dir = os.path.join(root, subset, point_dir_name)
        output_dir = os.path.join(root, subset, output_dir_name)
        if not os.path.isdir(mask_dir) or not os.path.isdir(point_dir):
            print(f'跳过目录: mask={mask_dir}, point={point_dir}')
            continue

        filter_connected_regions(mask_dir, point_dir, output_dir)
        manifest['splits'][subset] = {
            'mask_dir': mask_dir,
            'point_label_dir': point_dir,
            'domain_dir': output_dir,
            'file_count': len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        }

    manifest_path = write_manifest(root, manifest, output_dir_name)
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
    parser.add_argument('--mask_dir_name', type=str, default='mask')
    parser.add_argument('--point_dir_name', type=str, default='point_label_random')
    parser.add_argument('--output_dir_name', type=str, default='domain_random')
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        root = os.path.join(args.dataset_root, dataset)
        print(f'开始生成随机点标签对应连通域: {root}')
        process_dataset(
            root,
            args.splits,
            point_dir_name=args.point_dir_name,
            output_dir_name=args.output_dir_name,
            mask_dir_name=args.mask_dir_name
        )

if __name__ == '__main__':
    main()
