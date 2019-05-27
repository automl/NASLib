#!/usr/bin/env python3

"""
Translate a hyperparameter configuration as returned by the Auto-PyTorch .fit() method to a pytorch Sequential model. 
"""

import sys
import torch.nn as nn
from naslib.base.base_net import BaseNet

print("Python version", sys.version)
print("Starting test...")

example_hyperparameter_config = {'CreateDataLoader:batch_size': 125,
  'Imputation:strategy': 'median',
  'InitializationSelector:initialization_method': 'default',
  'InitializationSelector:initializer:initialize_bias': 'No',
  'LearningrateSchedulerSelector:lr_scheduler': 'cosine_annealing',
  'LossModuleSelector:loss_module': 'cross_entropy_weighted',
  'NetworkSelector:network': 'shapedresnet',
  'NormalizationStrategySelector:normalization_strategy': 'standardize',
  'OptimizerSelector:optimizer': 'sgd',
  'PreprocessorSelector:preprocessor': 'truncated_svd',
  'ResamplingStrategySelector:over_sampling_method': 'none',
  'ResamplingStrategySelector:target_size_strategy': 'none',
  'ResamplingStrategySelector:under_sampling_method': 'none',
  'TrainNode:batch_loss_computation_technique': 'standard',
  'LearningrateSchedulerSelector:cosine_annealing:T_max': 10,
  'LearningrateSchedulerSelector:cosine_annealing:T_mult': 2,
  'NetworkSelector:shapedresnet:activation': 'relu',
  'NetworkSelector:shapedresnet:blocks_per_group': 3,
  'NetworkSelector:shapedresnet:max_units': 368,
  'NetworkSelector:shapedresnet:num_groups': 1,
  'NetworkSelector:shapedresnet:resnet_shape': 'brick',
  'NetworkSelector:shapedresnet:use_dropout': 0,
  'NetworkSelector:shapedresnet:use_shake_drop': 0,
  'NetworkSelector:shapedresnet:use_shake_shake': 0,
  'OptimizerSelector:sgd:learning_rate': 0.022145527126351754,
  'OptimizerSelector:sgd:momentum': 0.321165154047931,
  'OptimizerSelector:sgd:weight_decay': 0.024578519182250506,
  'PreprocessorSelector:truncated_svd:target_dim': 100}

net = BaseNet(config=None,
              in_features=1,
              out_features=1,
              final_activation=None)

net.set_net_from_hyperpar_config(hyperpar_config=example_hyperparameter_config,
                                 in_features=1,
                                 out_features=1)
                                  

print(net.layers)
