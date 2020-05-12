import logging
import time

import torch
import torch.nn as nn

from naslib.optimizers.core import Evaluator
from naslib.utils import utils


class Searcher(Evaluator):
    def __init__(self, graph, parser, arch_optimizer, *args, **kwargs):
        super(Searcher, self).__init__(graph, parser, *args, **kwargs)
        self.arch_optimizer = arch_optimizer
        self.arch_optimizer.architectural_weights.to(self.device)
        self.run_kwargs['arch_optimizer'] = self.arch_optimizer

    def train(self, epoch, graph, optimizer, criterion, train_queue, valid_queue, *args, **kwargs):
        try:
            config = kwargs.get('config', graph.config)
            device = kwargs['device']
            arch_optimizer = kwargs['arch_optimizer']
        except Exception as e:
            raise ModuleNotFoundError('No configuration specified in graph or kwargs')

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        # Adjust arch optimizer for new search epoch
        arch_optimizer.new_epoch(epoch)

        start_time = time.time()
        for step, (input_train, target_train) in enumerate(train_queue):
            graph.train()
            n = input_train.size(0)

            input_train = input_train.to(device)
            target_train = target_train.to(device, non_blocking=True)

            # Architecture update
            arch_optimizer.forward_pass_adjustment()
            input_valid, target_valid = next(iter(valid_queue))
            input_valid = input_valid.to(device)
            target_valid = target_valid.to(device, non_blocking=True)

            arch_optimizer.step(graph, criterion, input_train, target_train, input_valid, target_valid, self.lr,
                                self.optimizer, config.unrolled)
            optimizer.zero_grad()

            # OP-weight update
            arch_optimizer.forward_pass_adjustment()
            logits = graph(input_train)
            loss = criterion(logits, target_train)
            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), config.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target_train, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % config.report_freq == 0:
                arch_key = list(arch_optimizer.architectural_weights.keys())[-1]
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        end_time = time.time()
        return top1.avg, objs.avg, end_time - start_time
