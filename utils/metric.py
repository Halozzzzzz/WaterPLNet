import cv2
import os
import numpy as np
import torch
import math


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def cal_fscore(hist):
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - np.diag(hist)
    FN = hist.sum(axis=1) - np.diag(hist)
    TN = hist.sum() - (FP + FN + TP)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f_score = 2 * precision * recall / (precision + recall)
    m_f_score = np.mean(f_score)
    return f_score, m_f_score


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def get_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self.get_hist(lp.flatten(), lt.flatten())
            # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # dice
        dice = 2 * np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        mdice = np.nanmean(dice)

        # -----------------其他指标------------------------------
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, dice, mdice, fwavacc


class IOUMetric_tensor:
    """
        Class to calculate mean-iou with tensor_type using fast_hist method
        """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = torch.zeros([num_classes, num_classes])

    def get_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = torch.bincount(
            self.num_classes * label_true[mask] +
            label_pred[mask], minlength=self.num_classes ** 2).view(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self.get_hist(lp.flatten(), lt.flatten())
            # miou
        iou = torch.diag(self.hist) / (self.hist.sum(dim=1) + self.hist.sum(dim=0) - torch.diag(self.hist))
        miou = torch.mean(iou)
        # dice
        dice = 2 * torch.diag(self.hist) / (self.hist.sum(dim=1) + self.hist.sum(dim=0))
        mdice = torch.mean(dice)

        # -----------------其他指标------------------------------
        # mean acc
        acc = torch.diag(self.hist).sum() / self.hist.sum()
        acc_cls = torch.mean(np.diag(self.hist) / self.hist.sum(dim=1))
        freq = self.hist.sum(dim=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, dice, mdice, fwavacc


def eval_hist(hist):
    # IoU & mIoU
    iou = cal_iou(hist)
    miou = np.nanmean(iou)

    # Overall Accuracy
    oa = cal_oa(hist)

    # Precision & mean Precision
    precision = cal_precision(hist)
    m_precision = np.mean(precision)

    # Recall & mean Recall
    recall = cal_recall(hist)
    m_recall = np.mean(recall)

    # F-score & mean F-score
    f_score = 2 * precision * recall / (precision + recall + 1e-8)
    m_f_score = np.mean(f_score)

    # Dice & mean Dice
    dice = cal_dice(hist)
    m_dice = np.mean(dice)

    # Kappa
    kappa = cal_kappa(hist)

    # FWIoU
    fwiou = cal_fwiou(hist)

    # 按照表格顺序返回
    return iou, miou, oa, precision, m_precision, recall, m_recall, f_score, m_f_score, dice, m_dice, kappa, fwiou


def get_tp_fp_tn_fn(hist):
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - np.diag(hist)
    FN = hist.sum(axis=1) - np.diag(hist)
    TN = hist.sum() - TP + FP + FN
    return TP, FP, FN, TN

def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def cal_recall(hist):
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    recall = TP / (TP + FN + 1e-8)
    return recall

def cal_precision(hist):
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    precision = TP / (TP + FP + 1e-8)
    return precision

def cal_iou(hist):
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    iou = TP / (TP + FN + FP + 1e-8)
    return iou

def cal_oa(hist):
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    oa = TP.sum() / (hist.sum() + 1e-8)
    return oa

def cal_dice(hist):
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    dice = 2 * TP / ((TP + FP) + (TP + FN) + 1e-8)
    return dice
        
def cal_fwiou(hist):
    N = hist.sum()
    TP, FP, FN, TN = get_tp_fp_tn_fn(hist)
    IoU = cal_iou(hist)
    freq = (TP + FN) / (N + 1e-8)
    fwiou = (freq * IoU).sum()
    return fwiou


def cls_accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    y_true = target.data.cpu().numpy()
    y_pred = output.data.cpu().numpy()

    # True Positive:即y_true与y_pred中同时为1的个数
    TP = np.sum(np.multiply(y_true, y_pred))

    # False Positive:即y_true中为0但是在y_pred中被识别为1的个数
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    # False Negative:即y_true中为1但是在y_pred中被识别为0的个数
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    # True Negative:即y_true与y_pred中同时为0的个数
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    # 根据上面得到的值计算A、P、R、F1
    kappa = (TP + TN) / (TP + FP + FN + TN)  # y_pred与y_ture中同时为1或0
    P = TP / (TP + FP)  # y_pred中为1的元素同时在y_true中也为1
    R = TP / (TP + FN)  # y_true中为1的元素同时在y_pred中也为1
    F1 = 2 * P * R / (P + R)

    return P, R, kappa, F1


def per_cls_accuracy(output, target):
    y_true = target.data.cpu().numpy()
    y_pred = output.data.cpu().numpy()
    acc_all = []
    for cls in range(y_true.shape[1]):
        true = np.sum(y_true[:, cls] == y_pred[:, cls])
        acc = true/y_true.shape[0]
        acc_all.append(acc)
    return acc_all


if __name__ == '__main__':
    a = torch.randint(0, 2, size=[30, 4])
    b = torch.randint(0, 2, size=[30, 4])

    P, R, kappa, F1 = cls_accuracy(a, b)
    print(P, R, kappa, F1)
    acc_all = per_cls_accuracy(a, b)
    print(acc_all)