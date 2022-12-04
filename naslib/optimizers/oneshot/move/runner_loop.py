import logging
from pyexpat import model
import sys
import naslib as nl
import random
import statistics as st
import os

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    OneShotNASOptimizer,
    RandomNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    GSparseOptimizer,
    MovementOptimizer,
    DrNASOptimizer,
    MovementOptimizer_test
)

from naslib.search_spaces import NasBench201SearchSpace, DartsSearchSpace, NasBench101SearchSpace, NATSBenchSizeSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from nas_201_api import NASBench201API as API #pip install nas-bench-201
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
api = API("/work/dlclarge2/agnihotr-ml/nas301_test_acc/NASLib/naslib/data/NAS-Bench-201-v1_1-096897.pth") #path to the API please refer to https://github.com/D-X-Y/NAS-Bench-201 for downloading

config = utils.get_config_from_args()
c10_acc = -1
c100_acc = -1
img_acc = -1
model_path = ''
architecture = ''
val_acc = -1
count = 0

run_c10, run_c100, run_img, run_val = [], [], [], []

random_seeds = random.sample(range(1,100), 5)
term1 = '/'+str(config.seed)

logger = setup_logger(config.save.replace(term1, '') + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

for random_seed in random_seeds:
    term2 = '/'+str(random_seed)
    config.save = config.save.replace(term1, term2)
    config.seed = random_seed
    utils.set_seed(random_seed)
    os.makedirs(config.save, exist_ok=True)
    
    
    utils.log_args(config)

    supported_optimizers = {
        "darts": DARTSOptimizer(config),
        "gdas": GDASOptimizer(config),
        "oneshot": OneShotNASOptimizer(config),
        "rsws": RandomNASOptimizer(config),
        "re": RegularizedEvolution(config),
        "rs": RandomSearch(config),
        "ls": RandomSearch(config),
        "bananas": Bananas(config),
        "bp": BasePredictor(config),
        "gsparsity": GSparseOptimizer(config),
        "movement": MovementOptimizer(config),
        "drnas": DrNASOptimizer(config),
        "movement_test" : MovementOptimizer_test(config)
    }
    
    if config.dataset =='cifar100':
        num_classes=100
    elif config.dataset=='ImageNet16-120':
        num_classes=120
    else:
        num_classes=10
    supported_search_space ={
        "nasbench201" : NasBench201SearchSpace(),#num_classes),
        "darts" : DartsSearchSpace(),#num_classes),
        "nasbench101" : NasBench101SearchSpace(),#num_classes)
        "natsbenchsize" : NATSBenchSizeSearchSpace(),
    }

    search_space = supported_search_space[config.search_space]
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    
    optimizer = supported_optimizers[config.optimizer]
    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, config, lightweight_output=True)
    trainer.search()
    
    trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY, api=api)
    
    run_c10.append(trainer.best_c10_acc)
    run_c100.append(trainer.best_c100_acc)
    run_img.append(trainer.best_img_acc)
    run_val.append(trainer.val_acc)

    #import ipdb;ipdb.set_trace()
    if trainer.val_acc == 90.93:
        count +=1

    if trainer.best_c10_acc > c10_acc:
        c10_acc = trainer.best_c10_acc
        c100_acc = trainer.best_c100_acc
        img_acc = trainer.best_img_acc
        val_acc = trainer.val_acc
        model_path = trainer.model_path
        architecture = trainer.architecture

logger.info('Final architecture from pool:\n' + architecture)
logger.info("FINAL TEST ACCURACIES: \n\t{}: {}\n\t{}: {}\n\t{}: {}\n\t{}: {}".format('cifar10', c10_acc, 'cifar100', c100_acc, 'ImageNet16-120', img_acc, 'VAL ACC CIFAR-10', val_acc))
logger.info("mean and std of the runs: \n\t{}: {} +/- {}\n\t{}: {} +/- {}\n\t{}: {} +/- {}\n\t{}: {} +/- {}".format('cifar10', st.mean(run_c10), st.stdev(run_c10), 'cifar100', st.mean(run_c100), st.stdev(run_c100), 'ImageNet16-120', st.mean(run_img), st.stdev(run_img), 'VAL ACC CIFAR-10', st.mean(run_val), st.stdev(run_val)))
logger.info("\nFound the best architecture {} times.\n".format(count))
logger.info("Final model saved at: " + model_path)
