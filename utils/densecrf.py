import numpy as np
import cv2  # 需要安装 opencv-python
import torch
import torch.nn.functional as F

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def get_crf(opt, mask, img):
    """
    替代 DenseCRF 的简化版本：
    使用高斯滤波平滑每个类别通道，然后取最大概率的类别。
    """
    # mask: H x W x num_classes (softmax 概率)
    # img: 原图 H x W x 3 (uint8)

    H, W, C = mask.shape
    smoothed = np.zeros_like(mask)

    for c in range(C):
        smoothed[..., c] = cv2.GaussianBlur(mask[..., c], (5, 5), sigmaX=3)

    label_map = np.argmax(smoothed, axis=-1)  # 取最大类别
    return label_map.astype(np.uint8)
