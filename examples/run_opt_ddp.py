import logging
from naslib.defaults.lightling_trainer import Trainer as t
from naslib.defaults.lightning_evaluator import Trainer as e
from naslib.defaults.lightling_dataloader import DataModule
from naslib.optimizers import DARTSOptimizer, ConfigurableOptimizer , GDASOptimizer, RandomSearch, DrNASOptimizer
from naslib.optimizers.oneshot.configurable.components import SNASSampler
from naslib.search_spaces import SimpleCellSearchSpace, DartsSearchSpace, AutoformerSearchSpace
import torchvision
from naslib.utils import set_seed, setup_logger, get_config_from_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torch
if __name__ == '__main__':
    config = get_config_from_args()  # use --help so see the options
    set_seed(config.seed)
    #logger = setup_logger(config.save + "/log.log")
    #logger.setLevel(logging.INFO)  # default DEBUG is very verbose
    search_space = AutoformerSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
    optimizer = ConfigurableOptimizer(config, arch_sampler=SNASSampler(2.5))
    optimizer.adapt_search_space(search_space)
    train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
    )
    test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
    )
    cifar10_dm = DataModule(dataset="CIFAR10", data_dir=".", split_train=True,
    cutout_length=0, batch_size=128, workers=0)
    cifar10_dm.setup()
    net_train = t(optimizer, config)
    trainer_search = Trainer(precision=16,gpus=-1,strategy="ddp_spawn",replace_sampler_ddp=True,max_epochs=10)
    trainer_search.fit(net_train,datamodule=cifar10_dm)
    cifar10_dm = DataModule(dataset="CIFAR10", data_dir=".", split_train=False,cutout_length=0, batch_size=128, workers=0)
    cifar10_dm.setup()
    net_eval = e(optimizer,config)
    trainer_eval = Trainer(precision=16,gpus=-1,strategy="ddp_spawn",max_epochs=10,replace_sampler_ddp=True)
    trainer_eval.fit(net_eval,datamodule=cifar10_dm)
    if trainer_eval.global_rank == 0:
        trainer_eval = Trainer(gpus=1)
        trainer_eval.test(net_eval,datamodule=cifar10_dm)