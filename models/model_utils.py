import torch
import torch.nn as nn

from models.backbones import ResNet50, ResNet101, ResNet34, ResNet18 

def build_backbone(backbone, output_stride, pretrained=False, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, pretrained=pretrained, in_c=in_c)
    elif backbone == 'resnet101':
        return ResNet101(output_stride, pretrained=pretrained, in_c=in_c)
    elif backbone == 'resnet34':
        return ResNet34(output_stride, pretrained=pretrained, in_c=in_c)
    elif backbone == 'resnet18':
        return ResNet18(output_stride, pretrained=pretrained, in_c=in_c)
    else:
        raise NotImplementedError


def build_channels(backbone):
    if backbone in ['resnet34', 'resnet18']:
        return [64, 128, 256, 512]
    else:
        return [256, 512, 1024, 2048]