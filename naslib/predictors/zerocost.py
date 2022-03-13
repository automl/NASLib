"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip
based on https://github.com/mohsaied/zero-cost-nas
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
from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.predictors.utils.pruners import predictive
from naslib.search_spaces.darts.conversions import convert_compact_to_genotype

logger = logging.getLogger(__name__)


class ZeroCost(Predictor):
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

            #TODO: unify these when graphs are fixed for every space
            if "nasbench201" in self.config.search_space:
                #NOTE: parsing the NASLib object creates the nn.Module.
                test_arch.prepare_discretization()
                test_arch.prepare_evaluation()
                test_arch.parse()

                network = test_arch
                logger.info('Parsed architecture')

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

            # set up loss function
            if self.config.dataset in ['class_object', 'class_scene']:
                loss_fn = SoftmaxCrossEntropyWithLogits()
            elif self.config.dataset == 'autoencoder':
                loss_fn = torch.nn.L1Loss()
            else:
                loss_fn = F.cross_entropy

            network = network.to(self.device)

            score = predictive.find_measures(
                network,
                self.train_loader,
                (self.dataload, self.num_imgs_or_batches, self.num_classes),
                self.device,
                loss_fn=loss_fn,
                measure_names=[self.method_type],
            )

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
