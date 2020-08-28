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


class Trainer(object):
    """
    Class which handles all the training.

    - Data loading and preparing batches
    - train loop
    - gather statistics
    - do the final evaluation
    """

    def __init__(self, optimizer, dataset, config, parser):
        self.optimizer = optimizer
        self.dataset = dataset
        self.epochs = 1     # config.epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._set_seed(config.seed)
        self._prepare_dataloaders(parser.get_train_val_loaders)


    @staticmethod
    def _set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            cudnn.benchmark = False
            cudnn.enabled = True
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(seed)


    def _prepare_dataloaders(self, get_data_loaders):
        train_queue, valid_queue, test_queue, train_transform, valid_transform = get_data_loaders()
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue
        self.train_transform = train_transform  # TODO they are not used are they?
        self.valid_transform = valid_transform


    def train(self):
        print("Start training")
        for e in range(self.epochs):
            self.optimizer.new_epoch(e)
            for step, (data_train, data_val) in enumerate(zip(self.train_queue, self.valid_queue)):
                # logits_train, logits_val, train_loss = 
                self.optimizer.step(data_train, data_val)
                break
        print("Training finished")
                

    def evaluate(self, retrain=False):
        print("Start evaluation")
        best_arch = self.optimizer.get_final_architecture()

        if retrain:
            best_arch.reset_weights(inplace=True)
            optim = self.optimizer.get_weight_optimizer()
            optim = optim(best_arch.parameters(), 0.01)
            
            # train from scratch
            for step, data_train in enumerate(self.train_queue):
                pass
        

        # measure final test accuracy
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        best_arch.eval()

        for data_test in self.test_queue:
            input_test, target_test = data_test
            n = input_test.size(0)

            with torch.no_grad():
                logits = best_arch(input_test)

                prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)

            break
        
        print("Evaluation finished. Test accuracies: top-1 = {}, top-5 = {}".format(top1.avg, top5.avg))









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
