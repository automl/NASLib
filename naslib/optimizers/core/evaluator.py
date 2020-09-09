import codecs
import time
import json
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from naslib.utils import utils
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)

class Trainer(object):
    """
    Class which handles all the training.

    - Data loading and preparing batches
    - train loop
    - gather statistics
    - do the final evaluation
    """

    def __init__(self, optimizer, dataset, config):
        self.optimizer = optimizer
        self.dataset = dataset
        self.config = config
        self.epochs = config.epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._prepare_dataloaders()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.op_optimizer, float(self.epochs), eta_min=self.config.learning_rate_min)

        self.train_top1 = utils.AvgrageMeter()
        self.train_top5 = utils.AvgrageMeter()
        self.val_top1 = utils.AvgrageMeter()
        self.val_top5 = utils.AvgrageMeter()



    def _prepare_dataloaders(self):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(self.config)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue


    def search(self):
        logger.info("Start training")
        self.optimizer.before_training()
        for e in range(self.epochs):
            self.optimizer.new_epoch(e)

            for step, (data_train, data_val) in enumerate(zip(self.train_queue, self.valid_queue)):
                data_train = (data_train[0].to(self.device), data_train[1].to(self.device, non_blocking=True))
                data_val = (data_val[0].to(self.device), data_val[1].to(self.device, non_blocking=True))

                stats = self.optimizer.step(data_train, data_val)
                logits_train, logits_val, train_loss, val_loss = stats

                self._store_accuracies(logits_train, data_train[1], 'train')
                self._store_accuracies(logits_val, data_val[1], 'val')
                
                log_every_n_seconds(logging.INFO, "Epoch {}-{}, Train loss: {:.5}, validation loss: {:.5}, learning rate: {}".format(
                    e, step, train_loss, val_loss, self.scheduler.get_last_lr()), n=5)
                
            self._log_and_reset_accuracies(e)
            self.scheduler.step()
        self.optimizer.after_training()
        logger.info("Training finished")
    

    def _log_and_reset_accuracies(self, epoch):
        logger.info("Epoch {} done. Train accuracy (top1, top5): {:.5}, {:.5}, Validation accuracy: {:.5}, {:.5}".format(
                epoch,
                self.train_top1.avg, self.train_top5.avg,
                self.val_top1.avg, self.val_top5.avg
            ))
        self.train_top1.reset()
        self.train_top5.reset()
        self.val_top1.reset()
        self.val_top5.reset()


    def _store_accuracies(self, logits, target, split):

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == 'train':
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == 'val':
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")


    def evaluate(self, retrain=True):
        logger.info("Start evaluation")

        best_arch = self.optimizer.get_final_architecture()
        logger.info("Final architecture:\n" + best_arch.modules_str())

        if retrain:
            best_arch.reset_weights(inplace=True)
            optim = self.optimizer.get_op_optimizer()
            optim = optim(
                best_arch.parameters(), 
                self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, float(self.epochs), eta_min=self.config.learning_rate_min)

            grad_clip = self.config.grad_clip
            loss = torch.nn.CrossEntropyLoss()

            best_arch.train()
            self.train_top1.reset()
            self.train_top5.reset()

            # train from scratch
            for e in range(self.epochs):
                for i, (input_train, target_train) in enumerate(self.train_queue):
                    input_train = input_train.to(self.device)
                    target_train = target_train.to(self.device, non_blocking=True)

                    optim.zero_grad()
                    logits_train = best_arch(input_train)
                    train_loss = loss(logits_train, target_train)
                    train_loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(best_arch.parameters(), grad_clip)
                    optim.step()

                    self._store_accuracies(logits_train, target_train, 'train')
                    log_every_n_seconds(logging.INFO, "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                        e, i, train_loss, scheduler.get_last_lr()), n=5)
                scheduler.step()

                logger.info("Epoch {} done. Train accuracy (top1, top5): {:.5}, {:.5}".format(e,
                    self.train_top1.avg, self.train_top5.avg))
                self.train_top1.reset()
                self.train_top5.reset()

        # measure final test accuracy
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

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
            
            log_every_n_seconds(logging.INFO, "Inference batch {} of {}.".format(i, len(self.test_queue)), n=5)

        logger.info("Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(top1.avg, top5.avg))















class Evaluator(object):
    """
    Class for training...?
    """
    def __init__(self, graph, parser, *args, **kwargs):
        self.graph = graph
        self.parser = parser
        try:
            self.config = kwargs.get('config', graph.config)
        except:
            raise Exception('No configuration specified in graph or kwargs')
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.manual_seed(self.config.seed)
            torch.cuda.set_device(self.config.gpu)
            cudnn.benchmark = False
            cudnn.enabled = True
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(self.config.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # dataloaders
        train_queue, valid_queue, test_queue, train_transform, valid_transform = parser.get_train_val_loaders()
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue
        self.train_transform = train_transform
        self.valid_transform = valid_transform

        criterion = eval('nn.' + self.config.criterion)()
        self.criterion = criterion.cuda()

        self.model = self.graph.to(self.device)

        n_parameters = utils.count_parameters_in_MB(self.model)
        logging.info("param size = %fMB", n_parameters)

        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     self.config.learning_rate,
        #     momentum=self.config.momentum,
        #     weight_decay=self.config.weight_decay)
        # self.optimizer = optimizer
        self.optimizer = None

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, float(self.config.epochs), eta_min=self.config.learning_rate_min)
        self.scheduler = None

        logging.info('Args: {}'.format(self.config))
        self.run_kwargs = {}

        self.errors_dict = utils.AttrDict(
            {'train_acc': [],
             'train_loss': [],
             'valid_acc': [],
             'valid_loss': [],
             'test_acc': [],
             'test_loss': [],
             'runtime': [],
             'params': n_parameters}
        )

    def run(self, *args, **kwargs):
        if 'epochs' not in kwargs:
            epochs = self.config.epochs
        else:
            raise ('No number of epochs specified to run network')

        for epoch in range(epochs):
            # self.lr = self.scheduler.get_last_lr()[0]
            self.lr = 0.01
            logging.info('epoch %d lr %e', epoch, self.lr)
            for n in self.graph.nodes:
                node = self.graph.get_node_op(n)    # subgraph
                if type(node).__name__ == 'Cell':
                    node.drop_path_prob = self.parser.config.drop_path_prob * epoch / epochs

            train_acc, train_obj, runtime = self.train(epoch, self.model, self.optimizer, self.criterion, self.train_queue,
                                                       self.valid_queue, device=self.device, **self.run_kwargs)
            logging.info('train_acc %f', train_acc)

            if len(self.valid_queue) != 0:
                valid_acc, valid_obj = self.infer(self.model, self.criterion, self.valid_queue, device=self.device)
                logging.info('valid_acc %f', valid_acc)
                self.errors_dict.valid_acc.append(valid_acc)
                self.errors_dict.valid_loss.append(valid_obj)

            test_acc, test_obj = self.infer(self.model, self.criterion,
                                            self.test_queue, device=self.device)
            logging.info('test_acc %f', test_acc)

            if hasattr(self.graph, 'query_architecture'):
                # Record anytime performance
                arch_info = self.graph.query_architecture(self.arch_optimizer.architectural_weights)
                logging.info('epoch {}, arch {}'.format(epoch, arch_info))
                if 'arch_eval' not in self.errors_dict:
                    self.errors_dict['arch_eval'] = []
                self.errors_dict['arch_eval'].append(arch_info)

            self.errors_dict.train_acc.append(train_acc)
            self.errors_dict.train_loss.append(train_obj)
            self.errors_dict.test_acc.append(test_acc)
            self.errors_dict.test_loss.append(test_obj)
            self.errors_dict.runtime.append(runtime)
            self.log_to_json(self.parser.config.save)
        Evaluator.save(self.parser.config.save, self.model, epoch)

    def train(self, epoch, graph, optimizer, criterion, train_queue, valid_queue, *args, **kwargs):
        try:
            config = kwargs.get('config', graph.config)
            device = kwargs['device']
        except Exception as e:
            raise ModuleNotFoundError('No configuration specified in graph or kwargs')

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        start_time = time.time()
        for step, (input, target) in enumerate(train_queue):
            graph.train()
            n = input.size(0)

            input = input.to(device)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            # logits, logits_aux = graph(input)
            logits = graph(input)
            loss = criterion(logits, target)
            # if config.auxiliary:
            #    loss_aux = criterion(logits_aux, target)
            #    loss += config.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), config.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % config.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        end_time = time.time()
        return top1.avg, objs.avg, end_time-start_time

    def infer(self, graph, criterion, valid_queue, *args, **kwargs):
        try:
            config = kwargs.get('config', graph.config)
            device = kwargs['device']
        except:
            raise ('No configuration specified in graph or kwargs')

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        graph.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = input.to(device)
                target = target.to(device, non_blocking=True)
                # logits, _ = graph(input)
                logits = graph(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)

                if step % config.report_freq == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    @staticmethod
    def save(save_path, model, epoch):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        utils.save(model, os.path.join(save_path, 'model_{}.pt'.format(epoch)))

    def log_to_json(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with codecs.open(os.path.join(save_path,
                                      'errors_{}.json'.format(self.config.seed)),
                         'w', encoding='utf-8') as file:
            json.dump(self.errors_dict, file, separators=(',', ':'))
