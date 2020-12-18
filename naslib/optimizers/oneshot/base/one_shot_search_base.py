import logging
import sys

import torch.nn as nn

from naslib.optimizers.evaluator import Evaluator
from naslib.utils import utils

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class OneShotSearchBase(Evaluator):
    def __init__(self, graph, arch_optimizer, *args, **kwargs):
        super(OneShotSearchBase, self).__init__(graph, *args, **kwargs)
        self.arch_optimizer = arch_optimizer
        self.arch_optimizer.architectural_weights.to(self.device)
        self.run_kwargs['arch_optimizer'] = self.arch_optimizer

    def train(self, graph, optimizer, criterion, train_queue, valid_queue, *args, **kwargs):
        try:
            config = kwargs.get('config', graph.config)
            device = kwargs['device']
            arch_optimizer = kwargs['arch_optimizer']
        except Exception as e:
            raise ModuleNotFoundError('No configuration specified in graph or kwargs')

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input_train, target_train) in enumerate(train_queue):
            graph.train()
            n = input_train.size(0)

            input_train = input_train.to(device)
            target_train = target_train.to(device, non_blocking=True)

            input_valid, target_valid = next(iter(valid_queue))
            input_valid = input_valid.to(device)
            target_valid = target_valid.to(device)

            arch_optimizer.step(graph, criterion, input_train, target_train, input_valid, target_valid, self.lr,
                                self.optimizer, config.unrolled)

            optimizer.zero_grad()
            logits = graph(input_train)
            loss = criterion(logits, target_train)

            '''
            if config.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += config.auxiliary_weight * loss_aux
            '''
            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), config.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target_train, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % config.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


if __name__ == '__main__':
    from naslib.search_spaces.darts import MacroGraph, PRIMITIVES
    from naslib.optimizers.optimizer import DARTSOptimizer
    from naslib.utils import config_parser

    one_shot_optimizer = DARTSOptimizer()
    config = config_parser('../../../configs/default.yaml')
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES
    )
    one_shot_optimizer.create_optimizer(**config)

    evaluator = OneShotSearchBase(search_space, arch_optimizer=one_shot_optimizer)
    evaluator.run()
