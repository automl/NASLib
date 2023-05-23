import codecs
import time
import json
import logging
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from copy import deepcopy
from fvcore.common.checkpoint import PeriodicCheckpointer

from naslib.search_spaces.core.query_metrics import Metric

from naslib import utils
from naslib.utils.log import log_every_n_seconds, log_first_n

from .additional_primitives import DropPathWrapper

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """

    def __init__(self, optimizer, config):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        self.optimizer = optimizer
        self.config = config
        self.epochs = self.config.search.epochs

        # preparations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # measuring stuff
        self.QUERYABLE = False
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_loss = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()

        n_parameters = optimizer.get_model_size()
        logger.info("param size = %fMB", n_parameters)
        self.errors_dict = utils.AttrDict(
            {
                "train_acc": [],
                "train_loss": [],
                "valid_acc": [],
                "valid_loss": [],
                "test_acc": [],
                "test_loss": [],
                "runtime": [],
                "arch_eval": [],
                "params": n_parameters,
            }
        )

    def search(self, resume_from=""):
        """
        Start the architecture search.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
        logger.info("Start training")
        self.optimizer.before_training()
        checkpoint_freq = self.config.search.checkpoint_freq
        if self.optimizer.using_step_function:
            self.scheduler = self.build_search_scheduler(
                self.optimizer.op_optimizer, self.config
            )

            start_epoch = self._setup_checkpointers(
                resume_from, period=checkpoint_freq, scheduler=self.scheduler
            )
        else:
            start_epoch = self._setup_checkpointers(resume_from, period=checkpoint_freq)

        self.train_queue, self.valid_queue, _ = self.build_search_dataloaders(
            self.config
        )

        for e in range(start_epoch, self.epochs):
            self.optimizer.new_epoch(e)

            start_time = time.time()
            if self.optimizer.using_step_function:
                for step, (data_train, data_val) in enumerate(
                    zip(self.train_queue, self.valid_queue)
                ):
                    data_train = (
                        data_train[0].to(self.device),
                        data_train[1].to(self.device, non_blocking=True),
                    )
                    data_val = (
                        data_val[0].to(self.device),
                        data_val[1].to(self.device, non_blocking=True),
                    )

                    stats = self.optimizer.step(data_train, data_val)
                    logits_train, logits_val, train_loss, val_loss = stats

                    self._store_accuracies(logits_train, data_train[1], "train")
                    self._store_accuracies(logits_val, data_val[1], "val")

                    log_every_n_seconds(
                        logging.INFO,
                        "Epoch {}-{}, Train loss: {:.5f}, validation loss: {:.5f}, learning rate: {}".format(
                            e, step, train_loss, val_loss, self.scheduler.get_last_lr()
                        ),
                        n=5,
                    )

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=3,
                        )

                    self.train_loss.update(float(train_loss.detach().cpu()))
                    self.val_loss.update(float(val_loss.detach().cpu()))

                self.scheduler.step()

                end_time = time.time()

                self.errors_dict.train_acc.append(self.train_top1.avg)
                self.errors_dict.train_loss.append(self.train_loss.avg)
                self.errors_dict.valid_acc.append(self.val_top1.avg)
                self.errors_dict.valid_loss.append(self.val_loss.avg)
                self.errors_dict.runtime.append(end_time - start_time)
            else:
                end_time = time.time()
                (
                    train_acc,
                    train_loss,
                    valid_acc,
                    valid_loss,
                ) = self.optimizer.train_statistics()
                self.errors_dict.train_acc.append(train_acc)
                self.errors_dict.train_loss.append(train_loss)
                self.errors_dict.valid_acc.append(valid_acc)
                self.errors_dict.valid_loss.append(valid_loss)
                self.errors_dict.runtime.append(end_time - start_time)
                self.train_top1.avg = train_acc
                self.val_top1.avg = valid_acc

            self.periodic_checkpointer.step(e)

            anytime_results = self.optimizer.test_statistics()
            if anytime_results:
                # record anytime performance
                self.errors_dict.arch_eval.append(anytime_results)
                log_every_n_seconds(
                    logging.INFO,
                    "Epoch {}, Anytime results: {}".format(e, anytime_results),
                    n=5,
                )

            self._log_to_json()
            self._log_and_reset_accuracies(e)

        self.optimizer.after_training()
        logger.info("Training finished")

    def main_worker(self, gpu, ngpus_per_node, args, search_model, best_arch):
        logger.info("Start evaluation")
        if not best_arch:
            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            self._setup_checkpointers(search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info("Final architecture:\n" + best_arch.modules_str())

        if best_arch.QUERYABLE:
            metric = Metric.TEST_ACCURACY
            result = best_arch.query(metric=metric, dataset=self.config.dataset)
            logger.info("Queried results ({}): {}".format(metric, result))
            self.QUERYABLE = True
            return

        best_arch.reset_weights(inplace=True)
        logger.info("Starting retraining from scratch")

        args.gpu = gpu
        if gpu is not None:
            logger.info("Use GPU: {} for training".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all processes
                args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )

        if not torch.cuda.is_available():
            logger.warning("Using CPU, this will be slow!")
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                best_arch.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                best_arch = torch.nn.parallel.DistributedDataParallel(
                    best_arch, device_ids=[args.gpu]
                )
            else:
                best_arch.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                best_arch = torch.nn.parallel.DistributedDataParallel(best_arch)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            best_arch = best_arch.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            best_arch = torch.nn.DataParallel(best_arch).cuda()

        cudnn.benchmark = True

        (
            self.train_queue,
            self.valid_queue,
            self.test_queue,
        ) = self.build_eval_dataloaders(self.config)

        optim = self.build_eval_optimizer(best_arch.parameters(), self.config)
        scheduler = self.build_eval_scheduler(optim, self.config)

        start_epoch = self._setup_checkpointers(
            args.resume_from,
            search=False,
            period=self.config.evaluation.checkpoint_freq,
            model=best_arch,  # checkpointables start here
            optim=optim,
            scheduler=scheduler,
        )

        grad_clip = self.config.evaluation.grad_clip
        loss = torch.nn.CrossEntropyLoss()

        best_arch.train()
        self.train_top1.reset()
        self.train_top5.reset()
        self.val_top1.reset()
        self.val_top5.reset()

        # Enable drop path
        if isinstance(best_arch, torch.nn.DataParallel):
            best_arch.module.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", DropPathWrapper(edge.data.op)
                ),
                scope=best_arch.module.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )
        else:
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", DropPathWrapper(edge.data.op)
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

        # train from scratch
        epochs = self.config.evaluation.epochs
        for e in range(start_epoch, epochs):
            # update drop path probability
            drop_path_prob = self.config.evaluation.drop_path_prob * e / epochs
            if isinstance(best_arch, torch.nn.DataParallel):
                best_arch.module.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "drop_path_prob", drop_path_prob
                    ),
                    scope=best_arch.module.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )
            else:
                best_arch.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "drop_path_prob", drop_path_prob
                    ),
                    scope=best_arch.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )

            # Train queue
            for i, (input_train, target_train) in enumerate(self.train_queue):
                input_train = input_train.to(self.device)
                target_train = target_train.to(self.device, non_blocking=True)

                optim.zero_grad()
                logits_train = best_arch(input_train)
                train_loss = loss(logits_train, target_train)
                if hasattr(best_arch, "auxilary_logits"):  # darts specific stuff
                    log_first_n(logging.INFO, "Auxiliary is used", n=10)
                    auxiliary_loss = loss(best_arch.auxilary_logits(), target_train)
                    train_loss += (
                        self.config.evaluation.auxiliary_weight * auxiliary_loss
                    )
                train_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(best_arch.parameters(), grad_clip)
                optim.step()

                self._store_accuracies(logits_train, target_train, "train")
                log_every_n_seconds(
                    logging.INFO,
                    "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                        e, i, train_loss, scheduler.get_last_lr()
                    ),
                    n=5,
                )

                if torch.cuda.is_available():
                    log_first_n(
                        logging.INFO,
                        "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                        n=3,
                    )

            # Validation queue
            if self.valid_queue:
                for i, (input_valid, target_valid) in enumerate(self.valid_queue):

                    input_valid = input_valid.to(self.device).float()
                    target_valid = target_valid.to(
                        self.device, non_blocking=True
                    ).float()

                    # just log the validation accuracy
                    logits_valid = best_arch(input_valid)
                    self._store_accuracies(logits_valid, target_valid, "val")

            scheduler.step()
            self.periodic_checkpointer.step(e)
            self._log_and_reset_accuracies(e)

    def evaluate(
        self,
        retrain=True,
        search_model="",
        resume_from="",
        best_arch=None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool): Reset the weights from the architecure search
            search_model (str): Path to checkpoint file that was created during
                search. If not provided, then try to load 'model_final.pth' from search
            resume_from (str): Resume retraining from the given checkpoint file.
            multi_gpu (bool): Distribute training on multiple gpus.
            best_arch: Parsed model you want to directly evaluate and ignore the final model
                from the optimizer.
        """

        # best_arch.to(self.device)
        self.config.evaluation.resume_from = resume_from
        if retrain:
            if self.config.gpu is not None:
                logger.warning(
                    "You have chosen a specific GPU. This will completely \
                    disable data parallelism."
                )

            if (
                self.config.evaluation.dist_url == "env://"
                and self.config.evaluation.world_size == -1
            ):
                self.config.evaluation.world_size = int(os.environ["WORLD_SIZE"])

            self.config.evaluation.distributed = (
                self.config.evaluation.world_size > 1
                or self.config.evaluation.multiprocessing_distributed
            )
            ngpus_per_node = torch.cuda.device_count()

            if self.config.evaluation.multiprocessing_distributed:
                # Since we have ngpus_per_node processes per node, the
                # total world_size needs to be adjusted
                self.config.evaluation.world_size = (
                    ngpus_per_node * self.config.evaluation.world_size
                )
                # Use torch.multiprocessing.spawn to launch distributed
                # processes: the main_worker process function
                mp.spawn(
                    self.main_worker,
                    nprocs=ngpus_per_node,
                    args=(
                        ngpus_per_node,
                        self.config.evaluation,
                        search_model,
                        best_arch,
                    ),
                )
            else:
                # Simply call main_worker function
                self.main_worker(
                    self.config.gpu,
                    ngpus_per_node,
                    self.config.evaluation,
                    search_model,
                    best_arch,
                )

        if not self.QUERYABLE:
            # Disable drop path
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", edge.data.op.get_embedded_ops()
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

            # measure final test accuracy
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            best_arch.eval()

            for i, data_test in enumerate(self.test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(self.device)
                target_test = target_test.to(self.device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                log_every_n_seconds(
                    logging.INFO,
                    "Inference batch {} of {}.".format(i, len(self.test_queue)),
                    n=5,
                )

            logger.info(
                "Evaluation finished. Test accuracies: top-1 = {:.5}, \
                        top-5 = {:.5}".format(
                    top1.avg, top5.avg
                )
            )

    @staticmethod
    def build_search_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="train"
        )
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="val"
        )
        return train_queue, valid_queue, test_queue

    @staticmethod
    def build_eval_optimizer(parameters, config):
        return torch.optim.SGD(
            parameters,
            lr=config.evaluation.learning_rate,
            momentum=config.evaluation.momentum,
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

    def _log_and_reset_accuracies(self, epoch):
        logger.info(
            "Epoch {} done. Train accuracy (top1, top5): {:.5f}, {:.5f}, \
            Validation accuracy: {:.5f}, {:.5f}".format(
                epoch,
                self.train_top1.avg,
                self.train_top5.avg,
                self.val_top1.avg,
                self.val_top5.avg,
            )
        )
        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()

    def _store_accuracies(self, logits, target, split):
        """Update the accuracy counters"""
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == "train":
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == "val":
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")

    def _prepare_dataloaders(self, config, mode="train"):
        """
        Prepare train, validation, and test dataloaders with the splits defined
        in the config.

        Args:
            config (AttrDict): config from config file.
        """
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode
        )
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def _setup_checkpointers(
        self, resume_from="", search=True, period=1, **add_checkpointables
    ):
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
            save_dir=self.config.save + "/search"
            if search
            else self.config.save + "/eval",
            **checkpointables
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=period,
            max_iter=self.config.search.epochs
            if search
            else self.config.evaluation.epochs,
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
        with codecs.open(
            os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(self.errors_dict, file, separators=(",", ":"))
