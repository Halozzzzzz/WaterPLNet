from models.model_utils import build_backbone, build_channels
from models.blocks.dual import DualBranchModule
import torch
import torch.nn as nn
import torch.nn.functional as F

def clean_mask(mask, cls_label):
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)


def get_penalty(predict, cls_label):
    # cls_label: (n, c)
    # predict: (n, c, h, w)
    n, c, h, w = predict.size()
    predict = torch.softmax(predict, dim=1)

    # if a patch does not contain label c,
    # then none of the pixels in this patch can be assigned to label c
    loss0 = - (1 - cls_label.view(n, c, 1, 1)) * torch.log(1 - predict + 1e-6)
    loss0 = torch.mean(torch.sum(loss0, dim=1))

    # if a patch has only one type, then the whole patch should be assigned to this type
    sum = (torch.sum(cls_label, dim=-1, keepdim=True) == 1)
    loss1 = - (sum * cls_label).view(n, c, 1, 1) * torch.log(predict + 1e-6)
    loss1 = torch.mean(torch.sum(loss1, dim=1))
    return loss0 + loss1

def get_shadowloss(predict, cls_label, shadow_label):
    n, c, h, w = predict.size()
    predict = torch.softmax(predict, dim=1)

    water_prob = predict[:, 1, :, :]
    has_water = cls_label[:, 1] > 0
    no_water_mask = (~has_water).float().view(n, 1, 1)
    shadow_label = shadow_label.clone()
    shadow_label[shadow_label == 255] = 0

    shadow_loss_map = - torch.log(1 - water_prob + 1e-6)
    shadow_loss_map = shadow_loss_map * shadow_label.squeeze(1)
    shadow_loss_map = shadow_loss_map * no_water_mask
    return shadow_loss_map.mean()


def get_numiter_dilations(flag):
    num_iter = 1
    # dilations = [[3], [3], [3], [3]]
    dilations = [[1, 3, 5, 7], [1, 3, 5], [3], [1]]
    return num_iter, dilations


class Net(nn.Module):
    def __init__(self, opt, flag='train'):
        super(Net, self).__init__()

        self.backbone = build_backbone(opt.backbone, output_stride=32)
        self.img_size = opt.img_size

        num_iter, dilations = get_numiter_dilations(flag)
        channels = build_channels(opt.backbone)
        key_channels = 128
        self.flag = flag

        self.DB_layer1 = DualBranchModule(channels[0], key_channels, dilations[0])
        self.DB_layer2 = DualBranchModule(channels[1], key_channels, dilations[1])
        self.DB_layer3 = DualBranchModule(channels[2], key_channels, dilations[2])
        self.DB_layer4 = DualBranchModule(channels[3], key_channels, dilations[3])


        self.last_conv = nn.Sequential(
            nn.Conv2d(4 * key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )

        self.DB_layer = DualBranchModule(key_channels, key_channels, [1])

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(key_channels, key_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(),
            nn.Conv2d(key_channels, opt.num_classes, kernel_size=3, padding=1),
        )


    def resize_out(self, output):
        if output.shape[-2:] != (self.img_size, self.img_size):
            output = F.interpolate(
                output,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=True
            )
        return output

    def upsample_cat(self, p1, p2, p3, p4):
        p2 = F.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=p1.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x):
        l1, l2, l3, l4 = self.backbone(x)

        f1, align_loss1 = self.DB_layer1(l1)
        f2, align_loss2 = self.DB_layer2(l2)
        f3, align_loss3 = self.DB_layer3(l3)
        f4, align_loss4 = self.DB_layer4(l4)

        p4 = f4
        p3 = F.interpolate(p4, size=f3.size()[2:], mode='bilinear', align_corners=True) + f3
        p2 = F.interpolate(p3, size=f2.size()[2:], mode='bilinear', align_corners=True) + f2
        p1 = F.interpolate(p2, size=f1.size()[2:], mode='bilinear', align_corners=True) + f1

        cat = self.upsample_cat(p1, p2, p3, p4)
        feat = self.last_conv(cat)
        feat, align_loss5 = self.DB_layer(feat)

        out = self.seg_decoder(feat)
        out = self.resize_out(out)

        align_loss = (align_loss1 + align_loss2 + align_loss3 + align_loss4 + align_loss5)/5
        if self.flag == 'test':
            return out
        else:
            return out, align_loss
    
    def forward_loss(self, x, label, cls_label, mask_label, shadow_label, current_epoch):
        coarse_mask, align_loss = self.forward(x)

        # get loss
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        penalty = get_penalty(coarse_mask, cls_label)
        seg_loss = criterion(coarse_mask, label)
        # mask_label is the low-scatter prior region I_pr used by LCS.
        # shadow_label stores the selected connected region I_sr generated by PSRE.
        shadow_loss = get_shadowloss(coarse_mask, cls_label, mask_label)
        
        # Avoid NaN when the PSRE target is fully ignored.
        if (shadow_label != 255).sum() == 0:
            psr_loss = torch.tensor(0.0, device=shadow_label.device)
        else:
            psr_loss = criterion(coarse_mask, shadow_label)

        return {
            'seg_loss': seg_loss,
            'penalty': penalty,
            'psr_loss': psr_loss,
            'shadow_loss': shadow_loss,
            'align_loss': align_loss
        }
