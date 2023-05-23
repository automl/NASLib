import math
from .net_ops.cell_ops import *


class SiameseDecoder(nn.Module):
    """Linear classifier for Siamese Model"""
    def __init__(self, in_dim, out_dim, num_pieces):
        super(SiameseDecoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim[0] * num_pieces, out_dim)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FFDecoder(nn.Module):
    """Linear classifier for classification"""
    def __init__(self, in_dim, out_dim):
        super(FFDecoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim[0], out_dim)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SegmentationDecoder(nn.Module):
    def __init__(self, in_dim, target_dim, target_num_channel=3, norm=nn.BatchNorm2d):
        """
        8-Layer Conv & Deconv Decoder for Segmentation task
        :param in_dim: input feature map dimension
        :param target_dim: label dimension
        :param target_num_channel: number of classes
        :param norm: normalization layer for decoder
        """
        super(SegmentationDecoder, self).__init__()
        in_channel, in_width = in_dim[0], in_dim[1]  # (512, 14)
        out_width = target_dim[0]  # 256
        num_upsample = int(math.log2(out_width / in_width))
        assert num_upsample in [2, 3, 4, 5, 6], f"invalid num_upsample: {num_upsample}"

        model = [ConvLayer(in_channel, 1024, 3, 1, 1, nn.LeakyReLU(0.2, False), norm)]

        tmp_in_C = 1024
        for i in range(6 - num_upsample):  # add non-upsampling layers
            model += [ConvLayer(tmp_in_C, tmp_in_C // 2, 3, 1, 1, nn.LeakyReLU(0.2, False), norm)]
            tmp_in_C = tmp_in_C // 2

        for i in range(num_upsample - 1):  # add upsampling layers
            model += [DeconvLayer(tmp_in_C, tmp_in_C // 2, 3, 2, 1, nn.LeakyReLU(0.2, False), norm)]
            tmp_in_C = tmp_in_C // 2

        model += [DeconvLayer(tmp_in_C, tmp_in_C, 3, 2, 1, nn.LeakyReLU(0.2, False), norm)]

        model += [ConvLayer(tmp_in_C, target_num_channel, 3, 1, 1, None, None)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GenerativeDecoder(nn.Module):
    def __init__(self, in_dim, target_dim, target_num_channel=3, norm=nn.BatchNorm2d):
        """
        8-Layer Conv & Deconv Decoder for Image Translation (Pix2Pix) task
        :param in_dim: input feature map dimension
        :param target_dim: label dimension
        :param target_num_channel: number of classes
        :param norm: normalization layer for decoder
        """
        super(GenerativeDecoder, self).__init__()

        in_channel, in_width = in_dim[0], in_dim[1]  # (512, 14)
        out_width = target_dim[0]  # 256
        num_upsample = int(math.log2(out_width / in_width))
        assert num_upsample in [2, 3, 4, 5, 6], f"invalid num_upsample: {num_upsample}"

        self.conv1 = ConvLayer(in_channel, 1024, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv2 = ConvLayer(1024, 1024, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        if num_upsample == 6:
            self.conv3 = DeconvLayer(1024, 512, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv3 = ConvLayer(1024, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv4 = ConvLayer(512, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        if num_upsample >= 5:
            self.conv5 = DeconvLayer(512, 256, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv5 = ConvLayer(512, 256, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv6 = ConvLayer(256, 128, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        if num_upsample >= 4:
            self.conv7 = DeconvLayer(128, 64, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv7 = ConvLayer(128, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv8 = ConvLayer(64, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        if num_upsample >= 3:
            self.conv9 = DeconvLayer(64, 32, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv9 = ConvLayer(64, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv10 = ConvLayer(32, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv11 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv12 = ConvLayer(16, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv13 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv14 = ConvLayer(16, target_num_channel, 3, 1, 1, nn.Tanh(), None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        return x


if __name__ == '__main__':
    net = GenerativeDecoder((512, 16, 16), (256, 256), target_num_channel=3).cuda()
    #summary(net, (512, 16, 16))
