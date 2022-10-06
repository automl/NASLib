import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

from .cell_micro import ResNetBasicblock, MicroCell
from ..net_ops.cell_ops import ReLUConvBN


class MacroNet(nn.Module):
    """Adapted from torchvision/models/resnet.py"""

    def __init__(self, net_code, structure='full', input_dim=(224, 224), num_classes=75):
        super(MacroNet, self).__init__()
        assert structure in ['full', 'drop_last', 'backbone'], 'unknown structrue: %s' % repr(structure)
        self.structure = structure
        self._read_net_code(net_code)
        self.inplanes = self.base_channel
        self.feature_dim = [input_dim[0] // 4, input_dim[1] // 4]

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.base_channel // 2, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.base_channel // 2, affine=True, track_running_stats=True),
            ReLUConvBN(self.base_channel // 2, self.base_channel, 3, 2, 1, 1, True, True)
        )

        self.layers = []
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)  # channel change: [2, 4]; stride change: [3, 4]
            target_channel = self.inplanes * 2 if layer_type % 2 == 0 else self.inplanes
            stride = 2 if layer_type > 2 else 1
            self.feature_dim = [self.feature_dim[0] // stride, self.feature_dim[1] // stride]
            layer = self._make_layer(self.cell, target_channel, 2, stride, True, True)
            self.add_module(f"layer{i}", layer)
            self.layers.append(f"layer{i}")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if structure in ['drop_last', 'full'] else None
        self.head = nn.Linear(self.inplanes, num_classes) if structure in ['full'] else None

        if structure == 'full':
            self.output_dim = (1, num_classes)
        elif structure == 'drop_last':
            self.output_dim = (self.inplanes, 1, 1)
        elif structure == 'backbone':
            self.output_dim = (self.inplanes, *self.feature_dim)
        else:
            raise ValueError

        self._kaiming_init()

    def forward(self, x):
        x = self.stem(x)

        for i, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        if self.structure in ['full', 'drop_last']:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        if self.structure == 'full':
            x = self.head(x)

        return x

    def _make_layer(self, cell, planes, num_blocks, stride=1, affine=True, track_running_stats=True):
        layers = [cell(self.micro_code, self.inplanes, planes, stride, affine, track_running_stats)]
        self.inplanes = planes * cell.expansion
        for _ in range(1, num_blocks):
            layers.append(cell(self.micro_code, self.inplanes, planes, 1, affine, track_running_stats))
        return nn.Sequential(*layers)

    def _read_net_code(self, net_code):
        net_code_list = net_code.split('-')
        self.base_channel = int(net_code_list[0])
        self.macro_code = net_code_list[1]
        if net_code_list[-1] == 'basic':
            self.micro_code = 'basic'
            self.cell = ResNetBasicblock
        else:
            self.micro_code = [''] + net_code_list[2].split('_')
            self.cell = MicroCell

    def _kaiming_init(self):
        # kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    net = MacroNet("64-41414-3_33_333", structure='backbone').cuda()
    print(net)
    summary(net, (3, 256, 256))
