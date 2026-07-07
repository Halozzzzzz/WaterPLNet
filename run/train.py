import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from dataset import *
from models import *
from options import *
from utils import *


def init_distributed_mode():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", str(local_rank)))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA/NCCL.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch_dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def resolve_device(opt, local_rank):
    if opt.device != "auto":
        return torch.device(opt.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    opt = Point_Options().parse()
    set_seed(opt.seed)

    rank_id, local_rank, world_size = init_distributed_mode()
    is_distributed = world_size > 1
    is_main_process = rank_id == 0
    device = resolve_device(opt, local_rank)
    pin_memory = opt.pin and device.type == "cuda"

    if is_main_process:
        if is_distributed:
            print(f"[INFO] Starting distributed training: rank={rank_id}, world_size={world_size}")
        else:
            print(f"[INFO] Starting single-process training on {device}")

    history = {"train": {}, "val": {}}

    log_path, checkpoint_path, _, _, _ = create_save_path(opt)
    train_txt_path, val_txt_path, _ = create_data_path(opt)
    logger, timestamp = create_logger(log_path) if is_main_process else (None, None)

    if is_main_process:
        for key, value in vars(opt).items():
            logger.info(f"{key:>16}: {value}")

    train_transform = transforms.Compose([
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])

    train_dataset = Dataset_point(opt, train_txt_path, flag="train", transform=train_transform)
    val_dataset = Dataset_point(opt, val_txt_path, flag="val", transform=None)
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=opt.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    model = build_model(opt, flag="train").to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
        )

    optimizer = optim.AdamW(model.parameters(), lr=opt.base_lr, amsgrad=True)
    best_metric = float("inf")

    for epoch in range(opt.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        time_start = time.time()

        train_metrics = train(opt, epoch, model, train_loader, optimizer, logger, is_main_process, device)

        if is_main_process:
            val_metrics = validate(opt, epoch, model, val_dataset, logger, device, pin_memory)

            for key, value in train_metrics.items():
                history["train"].setdefault(key, []).append(value)
            for key, value in val_metrics.items():
                history["val"].setdefault(key, []).append(value)

            val_loss = val_metrics.get("loss", float("inf"))
            logger.info(f"best_val_metric: {best_metric:.4f}, current_val_metric: {val_loss:.4f}")

            if val_loss < best_metric:
                logger.info(f"epoch:{epoch} Save to model_best_{timestamp}")
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(
                    {"state_dict": state_dict},
                    os.path.join(checkpoint_path, f"model_best_{timestamp}.pth"),
                )
                best_metric = val_loss
            logger.info(f"Epoch {epoch} Time {time.time() - time_start:.2f}s ----------------------\n")

    if is_main_process:
        save_loss_curve(history, log_path, timestamp, logger)

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def get_loss_weights(opt):
    return {
        "seg_loss": opt.seg_weight,
        "penalty": opt.penalty_weight,
        "psr_loss": opt.psr_weight,
        "shadow_loss": opt.shadow_weight,
        "align_loss": opt.align_weight,
        "energy_loss": opt.energy_weight,
        "ce_loss": 0.0,
        "ncg_loss": 0.0,
        "exp_loss": opt.exp_weight,
        "con_loss": opt.con_weight,
    }


def train(opt, epoch, model, loader, optimizer, logger, is_main_process, device):
    model.train()
    loss_weights = get_loss_weights(opt)
    meter_dict = {"loss": AverageMeter()}

    for batch_idx, batch in enumerate(loader):
        lr = adjust_learning_rate(optimizer, epoch, base_lr=opt.base_lr, decay_every=30)

        img, label, cls_label, mask_label, shadow_label, _ = batch
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        mask_label = mask_label.to(device, non_blocking=True)
        shadow_label = shadow_label.to(device, non_blocking=True)

        if hasattr(model, "module"):
            loss_dict = model.module.forward_loss(img, label, cls_label, mask_label, shadow_label, current_epoch=epoch)
        else:
            loss_dict = model.forward_loss(img, label, cls_label, mask_label, shadow_label, current_epoch=epoch)

        total_loss = 0.0
        for key, value in loss_dict.items():
            if value is None:
                continue
            total_loss += value * loss_weights.get(key, 1.0)
            meter_dict.setdefault(key, AverageMeter()).update(value.item(), img.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        meter_dict["loss"].update(total_loss.item(), img.size(0))

        if is_main_process and batch_idx == len(loader) - 1:
            log_metrics(logger, f"Epoch:{epoch} | LR:{lr:.2e} | Train : ", meter_dict)

    return {key: meter.avg for key, meter in meter_dict.items()}


def validate(opt, epoch, model, val_dataset, logger, device, pin_memory):
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=pin_memory,
    )
    loss_weights = get_loss_weights(opt)
    meter_dict = {"loss": AverageMeter()}

    with torch.no_grad():
        for batch in val_loader:
            img, label, cls_label, mask_label, shadow_label, _ = batch
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            cls_label = cls_label.to(device, non_blocking=True)
            mask_label = mask_label.to(device, non_blocking=True)
            shadow_label = shadow_label.to(device, non_blocking=True)

            if hasattr(model, "module"):
                loss_dict = model.module.forward_loss(img, label, cls_label, mask_label, shadow_label, current_epoch=epoch)
            else:
                loss_dict = model.forward_loss(img, label, cls_label, mask_label, shadow_label, current_epoch=epoch)

            total_loss = 0.0
            for key, value in loss_dict.items():
                if value is None:
                    continue
                total_loss += value * loss_weights.get(key, 1.0)
                meter_dict.setdefault(key, AverageMeter()).update(value.item(), img.size(0))

            meter_dict["loss"].update(total_loss.item(), img.size(0))

    log_metrics(logger, " VAL : ", meter_dict)
    return {key: meter.avg for key, meter in meter_dict.items()}


def log_metrics(logger, prefix, meter_dict):
    print_keys = [
        "loss",
        "seg_loss",
        "exp_loss",
        "con_loss",
        "ce_loss",
        "ncg_loss",
        "penalty",
        "psr_loss",
        "shadow_loss",
        "align_loss",
    ]
    log_str = prefix
    for key in print_keys:
        if key in meter_dict:
            log_str += f"{key}:{meter_dict[key].avg:.4f} | "
    logger.info(log_str)


def save_loss_curve(history, log_path, timestamp, logger):
    if not history["train"]:
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    def plot_on_ax(ax, data_dict, title_prefix):
        ax.set_title(f"{title_prefix} Loss Components")
        for name, values in data_dict.items():
            epochs = list(range(len(values)))
            if name == "loss":
                ax.plot(epochs, values, label=name, linewidth=2.5, color="black", linestyle="--")
            else:
                ax.plot(epochs, values, label=name, linewidth=1.5, alpha=0.8)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")

    plot_on_ax(axs[0], history["train"], "Training")
    if history["val"]:
        plot_on_ax(axs[1], history["val"], "Validation")

    plt.tight_layout()
    loss_curve_dir = os.path.join(os.path.dirname(log_path), "loss_curve")
    os.makedirs(loss_curve_dir, exist_ok=True)
    save_path = os.path.join(loss_curve_dir, f"{timestamp}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info(f"Loss curve saved to {save_path}")


if __name__ == "__main__":
    main()
