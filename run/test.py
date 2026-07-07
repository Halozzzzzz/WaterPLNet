import datetime
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import ttach as tta
from torch.utils.data import DataLoader
from tqdm import tqdm

from datafiles.color_dict import color_dict
from dataset import *
from models import *
from options import *
from utils import *
from utils import get_crf


def resolve_device(opt):
    if opt.device != "auto":
        return torch.device(opt.device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_checkpoint_path(checkpoint, checkpoint_dir):
    if checkpoint is None:
        raise ValueError("Please pass --checkpoint with a .pth file path or checkpoint filename.")

    ckpt_path = Path(checkpoint)
    if ckpt_path.is_file():
        return ckpt_path

    ckpt_path = Path(checkpoint_dir) / checkpoint.strip("/\\")
    if ckpt_path.is_file():
        return ckpt_path

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def main():
    opt = Point_Options().parse()
    device = resolve_device(opt)
    pin_memory = opt.pin and device.type == "cuda"

    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)
    print("Device:", device)

    _, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    _, _, test_txt_path = create_data_path(opt)
    ckpt_path = resolve_checkpoint_path(opt.checkpoint, checkpoint_path)
    results_path = os.path.join(predict_path, ckpt_path.stem)

    test_dataset = Dataset_point(opt, test_txt_path, flag="test", transform=None)
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(opt, flag="test")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")
    print(f"Loaded checkpoint: {ckpt_path}")

    compute_metric = IOUMetric(num_classes=opt.num_classes)
    hist = np.zeros([opt.num_classes, opt.num_classes])

    for batch in tqdm(loader):
        img, label, _, _, _, filename = batch
        with torch.no_grad():
            inputs = img.to(device, non_blocking=True)
            output = tta_model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        mask = F.softmax(output, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = Normalize_back(img_np, flag=opt.dataset)
        crf_out = get_crf(opt, mask, img_np.astype(np.uint8))

        save_pred_anno_numpy(crf_out, results_path, filename, dict=color_dict, flag=True)

        label_np = label.squeeze(0).data.cpu().numpy()
        hist += compute_metric.get_hist(crf_out, label_np)
        save_error_map(crf_out, label_np, results_path, filename[0])

    metrics = summarize_metrics(hist)
    write_results(metrics, predict_path, ckpt_path)


def save_error_map(prediction, label, results_path, filename):
    pred = (prediction > 0).astype(np.uint8)
    gt = (label > 0).astype(np.uint8)
    h, w = pred.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)
    vis[(pred == 1) & (gt == 1)] = [0, 0, 1]
    vis[(pred == 1) & (gt == 0)] = [1, 0, 0]
    vis[(pred == 0) & (gt == 1)] = [1, 1, 0]
    vis[(pred == 0) & (gt == 0)] = [0, 0, 0]

    vis_save_dir = os.path.join(results_path, "class_vis")
    os.makedirs(vis_save_dir, exist_ok=True)
    save_path = os.path.join(vis_save_dir, filename.replace(".png", "_class.png"))
    plt.imsave(save_path, vis)


def summarize_metrics(hist):
    iou, miou, oa, precision, _, recall, _, f_score, _, dice, _, kappa, fwiou = eval_hist(hist)
    water_index = 1
    metrics = {
        "water-F1": f_score[water_index],
        "water_recall": recall[water_index],
        "water_precision": precision[water_index],
        "OA": oa,
        "Kappa": kappa,
        "MIoU": miou,
        "FWIoU": fwiou,
        "water_IoU": iou[water_index],
        "water_Dice": dice[water_index],
    }

    for key, value in metrics.items():
        print(f"{key + ':':<25}{value:>10.4f}")
    return metrics


def write_results(metrics, predict_path, ckpt_path):
    test_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file = os.path.join(predict_path, "test_results.txt")
    with open(result_file, "a", encoding="utf-8") as file:
        file.write("=" * 50 + "\n")
        file.write(f"Test Time: {test_time}\n")
        file.write(f"Checkpoint: {ckpt_path}\n")
        file.write("-" * 50 + "\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")
        file.write("\n")

    print(f"Test results saved to {result_file}")


if __name__ == "__main__":
    main()
