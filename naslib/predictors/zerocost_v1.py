"""
Author: Robin Ru @ University of Oxford
This contains implementations of jacov and snip based on
https://github.com/BayesWatch/nas-without-training (jacov)
and https://github.com/gahaalt/SNIP-pruning (snip)
Note that zerocost_v2.py contains variants of jacov and snip implemented
in subsequent work. However, we find this version of jacov tends to perform
better.
"""


import numpy as np
import torch
import logging
import gc

from naslib.predictors.predictor import Predictor
from naslib.predictors.utils.build_nets import get_cell_based_tiny_net
from naslib.utils.utils import get_project_root, get_train_val_loaders
from naslib.predictors.utils.build_nets.build_darts_net import NetworkCIFAR
from naslib.search_spaces.darts.conversions import convert_compact_to_genotype

logger = logging.getLogger(__name__)

def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))

class ZeroCostV1(Predictor):

    def __init__(self, config, batch_size = 64, method_type='jacov'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.batch_size = batch_size
        self.method_type = method_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.data = "{}/data".format(get_project_root())
        self.config = config
        if method_type == 'jacov':
            self.num_classes = 1
        else:
            num_classes_dic = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}
            self.num_classes = num_classes_dic[self.config.dataset]

    def pre_process(self):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode='train')

    def query(self, xtest, info=None):

        test_set_scores = []
        count = 0
        for test_arch in xtest:
            count += 1
            logger.info('zero cost: {} of {}'.format(count, len(xtest)))
            if 'nasbench201' in self.config.search_space:
                ops_to_nb201 = {'AvgPool1x1': 'avg_pool_3x3', 'ReLUConvBN1x1': 'nor_conv_1x1',
                                'ReLUConvBN3x3': 'nor_conv_3x3', 'Identity': 'skip_connect', 'Zero': 'none',}
                # convert the naslib representation to nasbench201
                cell = test_arch.edges[2, 3].op
                edge_op_dict = {(i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges}
                op_edge_list = ['{}~{}'.format(edge_op_dict[(i, j)], i - 1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])]
                arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)
                arch_config = {'name': 'infer.tiny', 'C': 16, 'N':5, 'arch_str': arch_str,
                               'num_classes': self.num_classes}

                network = get_cell_based_tiny_net(arch_config)  # create the network from configuration

            elif 'darts' in self.config.search_space:
                test_genotype = convert_compact_to_genotype(test_arch.compact)
                arch_config = {'name': 'darts', 'C': 32, 'layers': 8, 'genotype': test_genotype,
                               'num_classes': self.num_classes, 'auxiliary': False}
                network = NetworkCIFAR(arch_config)

            data_iterator = iter(self.train_loader)
            x, target = next(data_iterator)
            x, target = x.to(self.device), target.to(self.device)

            network = network.to(self.device)

            if self.method_type == 'jacov':
                jacobs, labels = get_batch_jacobian(network, x, target)
                # print('done get jacobs')
                jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

                try:
                    score = eval_score(jacobs, labels)
                except Exception as e:
                    print(e)
                    score = -10e8

            elif self.method_type == 'snip':
                criterion = torch.nn.CrossEntropyLoss()
                network.zero_grad()
                _, y = network(x)
                loss = criterion(y, target)
                loss.backward()
                grads = [p.grad.detach().clone().abs() for p in network.parameters() if p.grad is not None]

                with torch.no_grad():
                    saliences = [(grad * weight).view(-1).abs() for weight, grad in zip(network.parameters(), grads)]
                    score = torch.sum(torch.cat(saliences)).cpu().numpy()
                    if hasattr(self, 'ss_type') and self.ss_type == 'darts':
                        score = -score

            test_set_scores.append(score)
            network, data_iterator, x, target, jacobs, labels = None, None, None, None, None, None
            torch.cuda.empty_cache()
            gc.collect()

        return np.array(test_set_scores)
