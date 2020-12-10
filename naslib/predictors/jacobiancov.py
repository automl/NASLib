import random

import numpy as np
import torch

from naslib.predictors.predictor import Predictor
from naslib.predictors.utils.build_nets import get_cell_based_tiny_net
from naslib.utils import utils
from naslib.utils.utils import get_project_root


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

class jacobian_cov(Predictor):

    def __init__(self, config, task_name='nas201_cifar10', batch_size = 256, seed=1):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.batch_size = batch_size
        self.task_name = task_name
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.data = "{}/data".format(get_project_root())
        self.config = config

    def pre_process(self):

        if 'nas201' in self.task_name:
            self.dataset = self.task_name.split('_')[1]
            # api_loc = './data/NAS-Bench-201-v1_1-096897.pth'
            # self.api = API(api_loc)
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

    def query(self, xtest, info=None):

        test_set_scores = []
        for test_arch in xtest:
            if 'nas201' in self.task_name:
                ops_to_nb201 = {'AvgPool1x1': 'avg_pool_3x3', 'ReLUConvBN1x1': 'nor_conv_1x1',
                                'ReLUConvBN3x3': 'nor_conv_3x3', 'Identity': 'skip_connect', 'Zero': 'none',}
                # convert the naslib representation to nasbench201
                cell = test_arch.edges[2, 3].op
                edge_op_dict = {(i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges}
                op_edge_list = ['{}~{}'.format(edge_op_dict[(i, j)], i - 1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])]
                arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)
                arch_config = {'name': 'infer.tiny', 'C': 16, 'N':5, 'arch_str': arch_str, 'num_classes': 1}
            data_iterator = iter(self.train_loader)
            x, target = next(data_iterator)
            x, target = x.to(self.device), target.to(self.device)

            network = get_cell_based_tiny_net(arch_config)  # create the network from configuration
            network = network.to(self.device)

            jacobs, labels = get_batch_jacobian(network, x, target)
            jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

            try:
                score = eval_score(jacobs, labels)
            except Exception as e:
                print(e)
                score = -10e8
            test_set_scores.append(score)
        return np.array(test_set_scores)
