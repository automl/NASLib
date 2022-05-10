from torch import nn


class GAN(nn.Module):
    """GAN model used for Pix2Pix tasks
    Adapted from https://github.com/phillipi/pix2pix
    """
    def __init__(self, encoder, decoder, discriminator):
        super(GAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def denormalize(imgs, mean, std):
        for i, (m, s) in enumerate(zip(mean, std)):
            imgs[:, i, :, :] = imgs[:, i, :, :] * s + m
        return imgs
