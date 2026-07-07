# WaterPLNet

Official code release for **WaterPLNet**, a PyTorch implementation for noise-robust weakly supervised SAR water-body segmentation with point-level annotations.

This public copy keeps only the core WaterPLNet code: preprocessing, dataset loaders, the WaterPLNet model, training, testing, metrics, and utility functions. Checkpoints, logs, prediction images, ablation exports, reproduced comparison networks, temporary scripts, and generated result folders are intentionally excluded.

## Repository Structure

```text
WaterPLNet/
|-- DataProcess/        # crop, mask generation, point labels, domain labels, split files
|-- datafiles/          # train/val/test filename lists for MSLCC, GF3_3m, and FUSAR
|-- dataset/            # PyTorch dataset loaders and augmentations
|-- models/             # WaterPLNet model, backbone, and dual-branch modules
|-- options/            # command-line options
|-- run/                # train/test entry points
|-- utils/              # metrics, visualization, normalization, and smoothing helpers
|-- requirements.txt
`-- README.md
```

## Installation

```bash
conda create -n waterplnet python=3.10 -y
conda activate waterplnet
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if you plan to train on GPU.

## Dataset Layout

Datasets are not included in this repository. Place data under:

```text
dataset/<DATASET>/
|-- train/
|   |-- img/
|   |-- label/
|   |-- point_label/
|   |-- mask/
|   `-- domain/
|-- val/
`-- test/
```

Supported dataset names in the default split files and normalization settings are `MSLCC`, `GF3_3m`, and `FUSAR`.

Label convention:

- `0`: non-water
- `1`: water
- `255`: ignored or unlabeled

Each split file in `datafiles/<DATASET>/` contains one image filename per line.

## Data Preparation

Generate low-backscatter candidate masks from raw SAR images:

```bash
python DataProcess/generate_mask.py \
  --input_dir raw/GF3_3m/sar \
  --mask_dir raw/GF3_3m/mask \
  --method otsu \
  --morph close
```

Crop full images, dense labels, and masks into patches:

```bash
python DataProcess/crop.py \
  --image_dir raw/GF3_3m/sar \
  --label_dir raw/GF3_3m/gts \
  --mask_dir raw/GF3_3m/mask \
  --save_root dataset/GF3_3m \
  --size 256 \
  --stride 128
```

Generate point labels and connected-domain labels:

```bash
python DataProcess/make_point_label.py \
  --datasets GF3_3m \
  --method center \
  --output_dir_name point_label

python DataProcess/generate_domain.py \
  --datasets GF3_3m \
  --point_dir_name point_label \
  --output_dir_name domain
```

Write split files:

```bash
python DataProcess/write_txt.py \
  --data_root dataset/GF3_3m \
  --out_dir datafiles/GF3_3m
```

If you add a new dataset, calculate its image statistics and update `get_mean_std()` in `utils/util.py`:

```bash
python DataProcess/calculate_mean\&std.py --img_dir dataset/GF3_3m/train/img
```

## Training

Single process:

```bash
python run/train.py \
  --dataset MSLCC \
  --model_name WaterPLNet \
  --annotation_mode original \
  --batch_size 64 \
  --num_epochs 150
```

Distributed training:

```bash
GPUS=0,1 bash run/run_distributed_train.sh \
  --dataset MSLCC \
  --annotation_mode original \
  --batch_size 64
```

Training outputs are created under `ckpt/<DATASET>/` and are ignored by git:

- `checkpoint/`: model weights
- `log/`: training logs
- `loss_curve/`: loss curves
- `predict_test/`: test predictions and metrics

## Testing

Pass either a checkpoint filename under `ckpt/<DATASET>/checkpoint/` or a full checkpoint path:

```bash
python run/test.py \
  --dataset MSLCC \
  --model_name WaterPLNet \
  --checkpoint model_best_xxxx.pth
```

Predicted masks, color visualizations, error maps, and metrics are saved under `ckpt/<DATASET>/predict_test/`.

## Main Options

- `--model_name`: `WaterPLNet`
- `--backbone`: ResNet backbone, default `resnet18`
- `--annotation_mode`: `original`, `random`, `noise`, or `center_noise`
- `--seg_weight`, `--penalty_weight`, `--psr_weight`, `--shadow_weight`, `--align_weight`: loss weights
- `--data_root`, `--data_inform_path`, `--save_path`: override repository-relative defaults

## Citation

If this code is useful for your research, please cite the corresponding paper. The BibTeX entry can be added here after publication.
