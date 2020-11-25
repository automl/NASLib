# Author: Yang Liu @ Abacus.ai
# This is an implementation of GBDT predictor for NAS from the paper:
# Luo, Renqian, et al. "Neural architecture search with gbdt." arXiv preprint arXiv:2007.04785 (2020).

import itertools
import os
import random
import sys
import numpy as np
import lightgbm as lgb


# Nasbench101 has 3 candidate operations + 1 output per node
one_hot_nasbench101 = [[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1]]

# Nasbench 201 cells have the same adjacency matrix
adjacency_matrix_nasbench201 = [[0,1,1,1],
                                [0,0,1,1],
                                [0,0,0,1],
                                [0,0,0,0]]

# 5 types of operations in Nasbench201                                
one_hot_nasbench201 = [[1,0,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,0,0,1]]

def convert_arch_to_seq_nasbench101(matrix, ops, max_n=7):
    """
    This function uses the original one hot encoding strategy from Luo et al. 2020
    """
    seq = []
    n = len(matrix)
    max_n=7
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            # zero pad at the end to generate fixed length one hot feature vector
            seq += [0 for i in range(col)]
            seq += [0,0,0,0]
        else:
            # one hot encoding for connections from previous layers
            for row in range(col):
                seq.append(matrix[row][col])
            # one hot encoding for operation type
            seq += one_hot_nasbench101[ops[col]]
            # if ops[col] == CONV1X1:
            #     seq += [1,0,0,0]
            # elif ops[col] == CONV3X3:
            #     seq += [0,1,0,0]
            # elif ops[col] == MAXPOOL3X3:
            #     seq += [0,0,1,0]
            # elif ops[col] == OUTPUT:
            #     seq += [0,0,0,1]
    assert len(seq) == (5+max_n+3)*(max_n-1)/2
    return seq

def convert_arch_to_seq_nasbench201(matrix, ops):
    seq = []
    for col in range(0, 6):
        # one hot encoding for operation type
        seq.extend(one_hot_nasbench201[ops[col]])

    return seq

# generate feature names (optional)
def get_feature_name_nasbench101():
    n = 7
    feature_name = []
    for col in range(1, n):
        for row in range(col):
            feature_name.append('node {} connect to node {}'.format(col+1, row+1))
        feature_name.append('node {} is conv 1x1'.format(col+1))
        feature_name.append('node {} is conv 3x3'.format(col+1))
        feature_name.append('node {} is max pool 3x3'.format(col+1))
        feature_name.append('node {} is output'.format(col+1))
    return feature_name

def get_feature_name_nasbench201():
    n = 6
    feature_name = []
    for col in range(1, n+1):
        feature_name.append('node {} is op1'.format(col+1))
        feature_name.append('node {} is op2'.format(col+1))
        feature_name.append('node {} is op3'.format(col+1))
        feature_name.append('node {} is op4'.format(col+1))
        feature_name.append('node {} is op5'.format(col+1))
    return feature_name

class GBDTPredictor:
    def get_model(self, params=None):
        if params is None:
            # default parameters used in Luo et al. 2020
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'min_data_in_leaf':5, # added by Colin White
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }
        return params

    def fit(self, xtrain, ytrain, 
            params=None,
            **kwargs):
        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        xtrain = np.array(xtrain)
        # convert to lgb dataset
        lgb_train = lgb.Dataset(xtrain, ((ytrain-self.mean)/self.std))
        # get model
        params = self.get_model()
        # get feature names (optional)
        feature_name = None #get_feature_name_nasbench201()
        # default 100 round of boosing from Luo et al. 2020
        self.model = lgb.train(params, lgb_train, feature_name=feature_name, num_boost_round=100)
        #gbm.save_model(os.path.join(args.output_dir, 'model.txt'))        
        train_pred = np.squeeze(self.model.predict(xtrain,num_iteration=self.model.best_iteration))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def predict(self, xtest):
        xtest = np.array(xtest)
        return np.squeeze(self.model.predict(xtest)) * self.std + self.mean
