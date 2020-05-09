import logging
import sys

import torch
import torch.nn as nn

from naslib.optimizers.core import Evaluator
from naslib.optimizers.oneshot.darts import DARTSOptimizer
from naslib.search_spaces.core.operations import OPS
from naslib.search_spaces.darts import PRIMITIVES, MacroGraph
#from naslib.search_spaces.nasbench_201.nasbench_201 import MacroGraph
#from naslib.search_spaces.nasbench_201.primitives import NAS_BENCH_201 as PRIMITIVES
#from naslib.search_spaces.nasbench_201.primitives import OPS as NASBENCH_201_OPS
from naslib.utils import config_parser
from naslib.utils import utils

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class Searcher(Evaluator):
    def __init__(self, graph, arch_optimizer, *args, **kwargs):
        super(Searcher, self).__init__(graph, *args, **kwargs)
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
                arch_key = list(arch_optimizer.architectural_weights.keys())[-1]
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                logging.info(
                    'train {}: {}'.format(arch_key, torch.softmax(arch_optimizer.architectural_weights[arch_key],
                                                                  dim=-1)))

        return top1.avg, objs.avg


if __name__ == '__main__':
    #config = config_parser('../../configs/search_spaces/nasbench_201.yaml')
    config = config_parser('../../configs/default.yaml')

    one_shot_optimizer = DARTSOptimizer.from_config(**config)
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES,
        ops_dict=OPS
    )
    one_shot_optimizer.init()

    searcher = Searcher(search_space, arch_optimizer=one_shot_optimizer)
    searcher.run()
