from torch import nn


class FeedForwardNet(nn.Module):
    """FeedForwardNet class used by classification and regression tasks"""

    def __init__(self, encoder, decoder):
        super(FeedForwardNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))