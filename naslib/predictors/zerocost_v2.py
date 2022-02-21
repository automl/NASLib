"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip 
based on https://github.com/mohsaied/zero-cost-nas
Note that zerocost_v1.py contains the original implementations
of jacov and snip. Particularly, the original jacov implementation tends to
perform better than the one in this file.
"""

import random
import numpy as np
import torch
import logging
import torch.nn.functional as F
import math

from naslib.search_spaces.transbench101.loss import SoftmaxCrossEntropyWithLogits
from naslib.predictors.predictor import Predictor
from naslib.utils.utils import get_project_root, get_train_val_loaders
from naslib.predictors.utils.models.build_darts_net import NetworkCIFAR
from naslib.predictors.utils.models import nasbench2 as nas201_arch
from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.predictors.utils.pruners import predictive
from naslib.predictors.utils.pruners.measures.model_stats import get_model_stats
from naslib.search_spaces.darts.conversions import convert_compact_to_genotype

logger = logging.getLogger(__name__)


class ZeroCostV2(Predictor):
    def __init__(self, config, batch_size=64, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.batch_size = batch_size
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.method_type = method_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.data = "{}/data".format(get_project_root())
        self.config = config
        num_classes_dic = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}
        self.num_classes = None
        if self.config.dataset in num_classes_dic:
            self.num_classes = num_classes_dic[self.config.dataset]

    def pre_process(self):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode="train")

    def query(self, xtest, info=None):

        test_set_scores = []
        count = 0
        for test_arch in xtest:
            count += 1
            logger.info("zero cost: {} of {}".format(count, len(xtest)))
            
            if "nasbench201" in self.config.search_space:
                ops_to_nb201 = {
                    "AvgPool1x1": "avg_pool_3x3",
                    "ReLUConvBN1x1": "nor_conv_1x1",
                    "ReLUConvBN3x3": "nor_conv_3x3",
                    "Identity": "skip_connect",
                    "Zero": "none",
                }
                # convert the naslib representation to nasbench201
                cell = test_arch.edges[2, 3].op
                edge_op_dict = {
                    (i, j): ops_to_nb201[cell.edges[i, j]["op"].get_op_name]
                    for i, j in cell.edges
                }
                op_edge_list = [
                    "{}~{}".format(edge_op_dict[(i, j)], i - 1)
                    for i, j in sorted(edge_op_dict, key=lambda x: x[1])
                ]
                arch_str = "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
                arch_config = {
                    "name": "infer.tiny",
                    "C": 16,
                    "N": 5,
                    "arch_str": arch_str,
                    "num_classes": self.num_classes,
                }
                network = nas201_arch.get_model_from_arch_str(
                    arch_str, self.num_classes
                )  # create the network from configuration
                # zero-cost-proxy author has the following checking lines (which I think might be optional)
                arch_str2 = nas201_arch.get_arch_str_from_model(network)
                if arch_str != arch_str2:
                    print(
                        f"Value Error: orig_arch={arch_str}, convert_arch={arch_str2}"
                    )
                    measure_score = -10e8
                    return measure_score
                
            

            elif "darts" in self.config.search_space:
                test_genotype = convert_compact_to_genotype(test_arch.compact)
                arch_config = {
                    "name": "darts",
                    "C": 32,
                    "layers": 8,
                    "genotype": test_genotype,
                    "num_classes": self.num_classes,
                    "auxiliary": False,
                }
                network = NetworkCIFAR(arch_config)

            elif "nasbench101" in self.config.search_space:
                spec = nasbench1_spec._ToModelSpec(
                    test_arch.spec["matrix"], test_arch.spec["ops"]
                )
                network = nas101_arch.Network(
                    spec,
                    stem_out=128,
                    num_stacks=3,
                    num_mods=3,
                    num_classes=self.num_classes,
                )
            else:
                """
                note: parsing the NASLib object creates the nn.Module.
                Technically we can do this for nb101 / 201 / darts as well,
                but are keeping the original code above for consistency.
                """
                test_arch.parse()
                network = test_arch
                logger.info('Parsed architecture')

            # set up loss function
            if self.config.dataset in ['class_object', 'class_scene']:
                loss_fn = SoftmaxCrossEntropyWithLogits()
            elif self.config.dataset == 'autoencoder':
                loss_fn = torch.nn.L1Loss()
            else:
                loss_fn = F.cross_entropy

            network = network.to(self.device)

            # todo: rearrange the if statements so that nb101/201/darts can use this code as well:
            try: # useful when launching bash scripts            
                if self.method_type in ['flops', 'params']:
                    """
                    This code is from
                    https://github.com/microsoft/archai/blob/5fc5e5aa63f3ac51a384b41e32eee4e7e5da2481/archai/common/trainer.py#L201
                    """
                    data_iterator = iter(self.train_loader)
                    x, target = next(data_iterator)
                    x_shape = list(x.shape)
                    x_shape[0] = 1 # to prevent overflow errors with large batch size we will use a batch size of 1
                    model_stats = get_model_stats(network, input_tensor_shape=x_shape, clone_model=True)

                    # important to do to avoid overflow
                    mega_flops = float(model_stats.Flops)/1e6
                    mega_params = float(model_stats.parameters)/1e6

                    if self.method_type == 'params':
                        score = mega_params
                    elif self.method_type == 'flops':
                        score = mega_flops

                else:

                    score = predictive.find_measures(
                        network,
                        self.train_loader,
                        (self.dataload, self.num_imgs_or_batches, self.num_classes),
                        self.device,
                        loss_fn=loss_fn,
                        measure_names=[self.method_type],
                    )
            except: # useful when launching bash scripts
                print('find_measures failed')
                score = -1e8

            # some of the values need to be flipped
            if math.isnan(score):
                score = -1e8

            if (
                "nasbench101" in self.config.search_space
                and self.method_type == "jacov"
            ):
                score = -score
            elif "darts" in self.config.search_space and self.method_type in [
                "fisher",
                "grad_norm",
                "synflow",
                "snip",
            ]:
                score = -score

            test_set_scores.append(score)
            torch.cuda.empty_cache()

        return np.array(test_set_scores)
