import torch.nn as nn

from .net_ops.cell_ops import ConvLayer


class Discriminator(nn.Module):
    def __init__(self, norm="spectral"):
        """
        Discriminator component for Pix2Pix tasks
        :param norm: ["batch": BN, "spectral": spectral norm for GAN]
        """
        super(Discriminator, self).__init__()
        if norm == "batch":
            norm = nn.BatchNorm2d
        elif norm == "spectral":
            norm = nn.utils.spectral_norm
        else:
            raise ValueError(f"{norm} is invalid!")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # input: [batch x 6 x 256 x 256]
        self.conv1 = ConvLayer(6, 64, 5, 4, 2, nn.LeakyReLU(0.2), norm)
        self.conv2 = ConvLayer(64, 128, 5, 4, 2, nn.LeakyReLU(0.2), norm)
        self.conv3 = ConvLayer(128, 256, 5, 4, 2, nn.LeakyReLU(0.2), norm)
        self.conv4 = ConvLayer(256, 256, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv5 = ConvLayer(256, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv6 = ConvLayer(512, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv7 = ConvLayer(512, 1, 3, 1, 1, None, None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        return x.flatten()
