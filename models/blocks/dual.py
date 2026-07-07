import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, s=1):
        super().__init__()
        pad = d if k == 3 else (k // 2) * d
        self.conv = nn.Conv2d(in_c, out_c, k, s, pad, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return y


class ResBlockBN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBNReLU(ch, ch, k=3, d=1)
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.c1(x)
        y2 = self.c2(y1)
        y  = x + y2
        out = self.act(y)
        return out


class RobustBranch(nn.Module):
    def __init__(self, channels: int, dilations=(2,)):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(ConvBNReLU(in_c=channels, out_c=channels, k=3, d=d, s=1))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class SensitiveBranch(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        assert depth >= 1
        self.stem = ConvBNReLU(in_c=channels, out_c=channels, k=3, d=1, s=1)
        self.blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self.blocks.append(ResBlockBN(channels))

    def forward(self, x):
        y = self.stem(x)
        for blk in self.blocks:
            y = blk(y)
        return y


class ShapeChannelAlign(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        y = self.proj(x)
        y = self.bn(y)
        return y


class CosineAlignLoss(nn.Module):
    # 每个空间位置像素计算cos相似度后平均
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, fs, fr):
        fs_n = F.normalize(fs, p=2, dim=1, eps=self.eps)
        fr_n = F.normalize(fr, p=2, dim=1, eps=self.eps)
        cos_map = (fs_n * fr_n).sum(dim=1)
        loss = (1.0 - cos_map).mean()
        return loss


class DualBranchModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 robust_dilations=(2,),
                 sensitive_depth: int = 3,
                 use_projection: bool = True,
                 gate_init_bias: float = -2.0):
        super().__init__()
        self.out_channels = out_channels
        self.noise_strength_r = 0.1
        self.noise_strength_s = 0.02

        if use_projection or (in_channels != out_channels):
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.proj = nn.Identity()

        self.robust    = RobustBranch(channels=out_channels, dilations=tuple(robust_dilations))
        self.sensitive = SensitiveBranch(channels=out_channels, depth=sensitive_depth)

        self.align_r = ShapeChannelAlign(in_c=out_channels, out_c=out_channels)
        self.align_s = ShapeChannelAlign(in_c=out_channels, out_c=out_channels)

        self.align_loss_fn = CosineAlignLoss()

        self.fuse = ConvBNReLU(2 * out_channels, out_channels, k=1, d=1, s=1)


    def forward(self, x):
        x_proj = self.proj(x)

        r = self.robust(x_proj)
        s = self.sensitive(x_proj)

        if self.training:
            r = r + torch.randn_like(r) * self.noise_strength_r
            s = s + torch.randn_like(s) * self.noise_strength_s

        r_aligned = self.align_r(r, s)
        s_aligned = self.align_s(s, r_aligned)

        if self.align_loss_fn is not None:
            align_loss = self.align_loss_fn(s_aligned, r_aligned)
        else:
            align_loss = torch.tensor(0.0, device=r.device)

        # concat
        x_cat = torch.cat([r_aligned, s_aligned], dim=1)
        y = self.fuse(x_cat)

        # y = x_proj
        # align_loss = torch.tensor(0.0, device=y.device)

        return y, align_loss
