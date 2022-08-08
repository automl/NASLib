import codecs
from naslib.search_spaces.core.graph import Graph
import time
import json
import logging
import os 
import copy
import torch
import numpy as np
from naslib.utils.utils import iter_flatten, AttrDict
from fvcore.common.checkpoint import PeriodicCheckpointer
from torch.nn import functional as F
from naslib.search_spaces.core.query_metrics import Metric
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from naslib.utils import utils
from naslib.utils.logging import log_every_n_seconds, log_first_n

from typing import Callable
from .additional_primitives import DropPathWrapper

logger = logging.getLogger(__name__)


class Trainer(pl.LightningModule):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """
    def __init__(self, optimizer, config, lightweight_output=False):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        super().__init__()
        self.automatic_optimization = False
        self.optimizer = optimizer
        self.config = config
        self.epochs = self.config.search.epochs
        self.lightweight_output = lightweight_output
        # preparations
        #self.device = torch.device(
        #    "cuda:0" if torch.cuda.is_available() else "cpu")
        print("Before Measuring")
        # measuring stuff
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_loss = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()

        n_parameters = optimizer.get_model_size()
        #logger.info("param size = %fMB", n_parameters)
        self.errors_dict = utils.AttrDict({
            "train_acc": [],
            "train_loss": [],
            "valid_acc": [],
            "valid_loss": [],
            "test_acc": [],
            "test_loss": [],
            "runtime": [],
            "train_time": [],
            "arch_eval": [],
            "params": n_parameters,
        })
        self.summary_writer = None
        #logger.info("Start evaluation")
        print("before getting best architecture")
        self.best_arch = self.optimizer.get_final_architecture()
        for u,v, edge_data in self.best_arch.edges.data():
            if not edge_data.is_final():
                edge = AttrDict(head=u, tail=v, data=edge_data)
                edge.data.set("discretize", True, shared=True)
        print(self.best_arch)
        #logger.info("Final architecture:\n" + self.best_arch.modules_str())
        self.optim = self.build_eval_optimizer(self.best_arch.parameters(),
                                                  self.config)
        self.scheduler = self.build_eval_scheduler(self.optim, self.config)
        self.start_epoch = self._setup_checkpointers(
                    "",
                    search=False,
                    period=self.config.evaluation.checkpoint_freq,
                    model=self.best_arch,  # checkpointables start here
                    optim=self.optim,
                    scheduler=self.scheduler,
                )
        self.grad_clip = self.config.evaluation.grad_clip
        print("before loss")


        self.epochs = self.config.evaluation.epochs
        self.test_top1 = utils.AverageMeter()
        self.test_top5 = utils.AverageMeter()
    
    def training_step(self,batch, batch_idx):
        """
        Start the architecture search. d

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
 
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        self.optim.zero_grad()
        input_train, target_train = batch
        #print(self.best_arch)
        logits_train = self.best_arch(input_train)
        train_loss =  F.cross_entropy(logits_train, target_train)
        if hasattr(self.best_arch,"auxilary_logits"):  # darts specific stuff
            log_first_n(logging.INFO,
            "Auxiliary is used",
            n=10)
            auxiliary_loss = F.cross_entropy(self.best_arch.auxilary_logits(),
                                                  target_train)
            train_loss += (self.config.evaluation.auxiliary_weight *auxiliary_loss)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                                self.best_arch.parameters(), self.grad_clip)
        self.optim.step()
        self._store_accuracies(logits_train, target_train,
                                               "train")
        log_every_n_seconds(
                            logging.INFO,
                            "Epoch {}-{}, Train loss: {:.5}, learning rate: {}"
                            .format(self.current_epoch, self.global_step, train_loss, self.scheduler.get_last_lr()),
                            n=5,
                        )

    def validation_step(self, batch, batch_idx):
        # Validation queue
        self.best_arch.eval()
        input_valid, target_valid = batch
        with torch.no_grad():
            logits_valid = self.best_arch(input_valid)
            self._store_accuracies(logits_valid,target_valid, "val")

    def training_epoch_end(self, outputs) -> None:
        if torch.cuda.is_available():
                log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(
                                torch.cuda.memory_summary()),
                            n=20,
                        )
        self.scheduler.step()
        self.periodic_checkpointer.step(self.current_epoch)
        self._log_and_reset_accuracies(self.current_epoch)

        # measure final test accuracy



    def configure_optimizers(self):
        return self.optim


    def test_step(self,batch,batch_idx):


            self.best_arch.eval()
            input_test, target_test = batch
            n = input_test.size(0)
            with torch.no_grad():
                    logits = self.best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits,
                                                  target_test,
                                                  topk=(1, 5))
                    self.test_top1.update(prec1.data.item(),n)
                    self.test_top5.update(prec5.data.item(),n)

    def test_epoch_end(self,outputs):
            print(self.test_top1.avg, self.test_top5.avg)

    @staticmethod
    def build_search_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="train")
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="val")
        return train_queue, valid_queue, test_queue

    @staticmethod
    def build_eval_optimizer(parameters, config):
        return torch.optim.AdamW(
            parameters,
            lr=config.evaluation.learning_rate,
            weight_decay=config.evaluation.weight_decay,
        )

    @staticmethod
    def build_search_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.search.epochs,
            eta_min=config.search.learning_rate_min,
        )

    @staticmethod
    def build_eval_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.evaluation.epochs,
            eta_min=config.evaluation.learning_rate_min,
        )

    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train accuracy (top1, top5): {:.5f}, {:.5f}, Validation accuracy: {:.5f}, {:.5f}"
            .format(
                epoch,
                self.train_top1.avg,
                self.train_top5.avg,
                self.val_top1.avg,
                self.val_top5.avg,
            ))

        if writer is not None:
            writer.add_scalar('Train accuracy (top 1)', self.train_top1.avg,
                              epoch)
            writer.add_scalar('Train accuracy (top 5)', self.train_top5.avg,
                              epoch)
            writer.add_scalar('Train loss', self.train_loss.avg, epoch)
            writer.add_scalar('Validation accuracy (top 1)', self.val_top1.avg,
                              epoch)
            writer.add_scalar('Validation accuracy (top 5)', self.val_top5.avg,
                              epoch)
            writer.add_scalar('Validation loss', self.val_loss.avg, epoch)

        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()

    def _store_accuracies(self, logits, target, split):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == "train":
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == "val":
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError(
                "Unknown split: {}. Expected either 'train' or 'val'")

    def _prepare_dataloaders(self, config, mode="train"):
        """
        Prepare train, validation, and test dataloaders with the splits defined
        in the config.

        Args:
            config (AttrDict): config from config file.
        """
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def _setup_checkpointers(self,
                             resume_from="",
                             search=True,
                             period=1,
                             **add_checkpointables):
        """
        Sets up a periodic chechkpointer which can be used to save checkpoints
        at every epoch. It will call optimizer's `get_checkpointables()` as objects
        to store.

        Args:
            resume_from (str): A checkpoint file to resume the search or evaluation from.
            search (bool): Whether search or evaluation phase is checkpointed. This is required
                because the files are in different folders to not be overridden
            add_checkpointables (object): Additional things to checkpoint together with the
                optimizer's checkpointables.
        """
        checkpointables = self.optimizer.get_checkpointables()
        checkpointables.update(add_checkpointables)

        checkpointer = utils.Checkpointer(
            model=checkpointables.pop("model"),
            save_dir=self.config.save +
            "/search" if search else self.config.save + "/eval",
            # **checkpointables #NOTE: this is throwing an Error
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=period,
            max_iter=self.config.search.epochs
            if search else self.config.evaluation.epochs,
        )

        if resume_from:
            logger.info("loading model from file {}".format(resume_from))
            checkpoint = checkpointer.resume_or_load(resume_from, resume=True)
            if checkpointer.has_checkpoint():
                return checkpoint.get("iteration", -1) + 1
        return 0

    def _log_to_json(self):
        """log training statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        if not self.lightweight_output:
            with codecs.open(os.path.join(self.config.save, "errors.json"),
                             "w",
                             encoding="utf-8") as file:
                json.dump(self.errors_dict, file, separators=(",", ":"))
        else:
            with codecs.open(os.path.join(self.config.save, "errors.json"),
                             "w",
                             encoding="utf-8") as file:
                lightweight_dict = copy.deepcopy(self.errors_dict)
                for key in [
                        "arch_eval", "train_loss", "valid_loss", "test_loss"
                ]:
                    lightweight_dict.pop(key)
                json.dump([self.config, lightweight_dict],
                          file,
                          separators=(",", ":"))