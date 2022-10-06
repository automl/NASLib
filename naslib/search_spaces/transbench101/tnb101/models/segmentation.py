from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Segmentation(nn.Module):
    """Segmentation used by semanticsegment task"""
    def __init__(self, encoder, decoder):
        super(Segmentation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to_device(self, device_list, rank=None, ddp=False):
        self.device_list = device_list
        if len(self.device_list) > 1:
            if ddp:
                self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
                self.encoder = DDP(self.encoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.decoder = DDP(self.decoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.rank = rank
            else:
                self.encoder = nn.DataParallel(self.encoder).to(self.device_list[0])
                self.decoder = nn.DataParallel(self.decoder).to(self.device_list[0])
                self.rank = rank
        else:
            self.rank = self.device_list[0]
            self.to(self.rank)
