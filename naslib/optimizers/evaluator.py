import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable

from naslib.utils import utils


class Evaluator(object):
    def __init__(self, graph, *args, **kwargs):
        self.graph = graph
        try:
            self.config = config if 'config' in kwargs else graph.config
        except:
            raise('No configuration specified in graph or kwargs')


    @staticmethod
    def train(graph, optimizer, criterion, train_queue, *args, **kwargs):
        try:
            config = graph.config if 'config' not in kwargs
        except:
            raise('No configuration specified in graph or kwargs')

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            graph.train()
            n = input.size(0)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            #if architect in kwargs:
                # get a minibatch from the search queue with replacement
            #    input_search, target_search = next(iter(valid_queue))

            #    input_search = input_search.cuda()
            #    target_search = target_search.cuda(non_blocking=True)

            # Allow for warm starting of the one-shot model for more reliable architecture updates.
            #    if epoch >= self.args.warm_start_epochs:
            #        architect.step(input_train=input,
            #                       target_train=target,
            #                       input_valid=input_search,
            #                       target_valid=target_search,
            #                       eta=lr,
            #                       network_optimizer=self.optimizer,
            #                       unrolled=self.args.unrolled)

            optimizer.zero_grad()
            logits, logits_aux = graph(input)
            loss = criterion(logits, target)
            if config.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += config.auxiliary_weight*loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), config.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % config.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    @staticmethod
    def infer(self, graph, criterion, valid_queue):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        graph.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda(non_blocking=True)

                logits, _ = graph(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)

                if step % config.report_freq == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def save(self, epoch):
        utils.save(self.model, os.path.join(self.args.save, 'one_shot_model_{}.pt'.format(epoch)))

