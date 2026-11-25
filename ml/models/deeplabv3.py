"""DeepLabV3+ (backbone ResNet-50, torchvision) adapté aux entrées 2 canaux VV/VH.

torchvision fournit deeplabv3_resnet50 (ASPP + decoder). On adapte la première
convolution du stem à `in_channels` et on ajoute une tête de classification
aux classes souhaitées.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 3,
                 pretrained_backbone: bool = False):
        super().__init__()
        try:
            weights = "DEFAULT" if pretrained_backbone else None
            self.net = deeplabv3_resnet50(
                weights=None, weights_backbone=weights, num_classes=num_classes
            )
        except TypeError:
            self.net = deeplabv3_resnet50(
                pretrained=pretrained_backbone, pretrained_backbone=None,
                num_classes=num_classes
            )

        first = self.net.backbone.conv1
        if first.in_channels != in_channels:
            self.net.backbone.conv1 = nn.Conv2d(
                in_channels, first.out_channels,
                kernel_size=first.kernel_size, stride=first.stride,
                padding=first.padding, bias=False,
            )

    def forward(self, x):
        return self.net(x)["out"]
