import os
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def check_dir(path):
    """检查并创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已创建文件夹: {path}")

def infer_dataset_name(path):
    parts = os.path.normpath(path).split(os.sep)
    for name in ('GF3_3m', 'MSLCC', 'FUSAR'):
        if name in parts:
            return name
    return None


def get_default_percentile(dataset_name):
    # These values reproduce the current dataset masks more closely than full-image Otsu:
    # mostly white background, with only the darkest low-backscatter candidates selected.
    defaults = {
        'GF3_3m': 10.0,
        'MSLCC': 11.0,
        'FUSAR': 10.0,
    }
    return defaults.get(dataset_name, 15.0)


def read_sar_gray(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype not in (np.uint8, np.uint16):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def build_low_scatter_mask(img, method='percentile', percentile=15.0,
                           morph='close_open', kernel_size=3,
                           close_kernel_size=5, open_kernel_size=3):
    if method == 'otsu':
        max_value = 65535 if img.dtype == np.uint16 else 255
        threshold, binary_water = cv2.threshold(
            img,
            0,
            max_value,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        binary_water = (binary_water == max_value).astype(np.uint8)
    elif method == 'percentile':
        threshold = float(np.percentile(img, percentile))
        binary_water = (img <= threshold).astype(np.uint8)
    else:
        raise ValueError(f'不支持的 mask 生成方法: {method}')

    if morph != 'none':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if morph == 'open':
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_OPEN, kernel)
        elif morph == 'close':
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_CLOSE, kernel)
        elif morph == 'open_close':
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_OPEN, kernel)
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_CLOSE, kernel)
        elif morph == 'close_open':
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (close_kernel_size, close_kernel_size)
            )
            open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (open_kernel_size, open_kernel_size)
            )
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_CLOSE, close_kernel)
            binary_water = cv2.morphologyEx(binary_water, cv2.MORPH_OPEN, open_kernel)
        else:
            raise ValueError(f'不支持的形态学操作: {morph}')

    final_mask = np.full(binary_water.shape, 255, dtype=np.uint8)
    final_mask[binary_water > 0] = 1
    return final_mask, threshold


def generate_mask(input_dir, mask_dir, method='percentile', percentile=None,
                  morph='close_open', kernel_size=3,
                  close_kernel_size=5, open_kernel_size=3):
    """
    自动生成 SAR 低散射候选水体掩码。

    输出值保持项目原有约定：
    - 1: 低散射候选水体，显示为近黑色
    - 255: 背景，显示为白色
    """
    check_dir(mask_dir)

    valid_ext = ('.png', '.tif', '.tiff', '.jpg')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    if percentile is None:
        percentile = get_default_percentile(infer_dataset_name(input_dir))

    print(f"-> 在 {input_dir} 中找到 {len(file_list)} 张图片，开始处理...")
    print(
        f"-> method={method}, percentile={percentile}, morph={morph}, "
        f"kernel_size={kernel_size}, close_kernel_size={close_kernel_size}, "
        f"open_kernel_size={open_kernel_size}"
    )

    for fname in tqdm(file_list, desc="Mask Processing", ncols=80):
        img_path = os.path.join(input_dir, fname)
        save_path = os.path.join(mask_dir, fname)
        img = read_sar_gray(img_path)
        
        if img is None:
            print(f"无法读取图片: {fname}，跳过。")
            continue

        final_mask, _ = build_low_scatter_mask(
            img,
            method=method,
            percentile=percentile,
            morph=morph,
            kernel_size=kernel_size,
            close_kernel_size=close_kernel_size,
            open_kernel_size=open_kernel_size
        )
        Image.fromarray(final_mask).save(save_path)

    print(f"\n处理完成！结果已保存至: {mask_dir}")


def generate_otsu_mask(input_dir, mask_dir):
    generate_mask(
        input_dir,
        mask_dir,
        method='otsu',
        percentile=None,
        morph='close_open',
        close_kernel_size=5,
        open_kernel_size=3
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='SAR 原图路径'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        required=True,
        help='输出 mask 路径'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='otsu',
        choices=['percentile', 'otsu'],
        help='percentile 更贴近当前数据集 mask；otsu 为原大津法'
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=None,
        help='低散射分位数阈值；不指定时按数据集自动选择'
    )
    parser.add_argument(
        '--morph',
        type=str,
        default='close',
        choices=['none', 'open', 'close', 'open_close', 'close_open'],
        help='形态学操作'
    )
    parser.add_argument('--kernel_size', type=int, default=3, help='单步形态学核大小')
    parser.add_argument('--close_kernel_size', type=int, default=5, help='close_open 中 close 的核大小')
    parser.add_argument('--open_kernel_size', type=int, default=3, help='close_open 中 open 的核大小')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_mask(
        args.input_dir,
        args.mask_dir,
        method=args.method,
        percentile=args.percentile,
        morph=args.morph,
        kernel_size=args.kernel_size,
        close_kernel_size=args.close_kernel_size,
        open_kernel_size=args.open_kernel_size
    )

if __name__ == '__main__':
    main()
