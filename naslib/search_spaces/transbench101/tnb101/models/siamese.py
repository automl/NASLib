import torch
from torch import nn


class SiameseNet(nn.Module):
    """SiameseNet used in Jigsaw task"""
    def __init__(self, encoder, decoder):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 4:
            assert x.shape == (1, 3, 720, 1080)
            x = image2tiles4testing(x)
        imgtile_num = x.shape[1]
        encoder_output = []
        for index in range(imgtile_num):
            input_i = x[:, index, :, :, :]
            ith_encoder_output = self.encoder(input_i)
            encoder_output.append(ith_encoder_output)
        concat_output = torch.cat(encoder_output, axis=1)
        final_output = self.decoder(concat_output)
        return final_output


def image2tiles4testing(img, num_pieces=9):
    """
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img (tensor): Image to be cropped (1, 3, 720, 1080)
h
    Return:
    -----------
        img_tiles: tensor (1, 9, 3, 240, 360)
    """

    if num_pieces != 9:
        raise ValueError(f'Target permutation of Jigsaw is supposed to have length 9, getting {num_pieces} here')

    Ba, Ch, He, Wi = img.shape  # (1, 3, 720, 1080)

    unitH = int(He / 3)  # 240
    unitW = int(Wi / 3)  # 360

    return img.view(Ba, 9, Ch, unitH, unitW)
