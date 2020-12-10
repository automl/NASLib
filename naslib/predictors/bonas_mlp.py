# Author: Yang Liu @ Abacus.ai
# This is an implementation of gcn predictor for NAS from the paper:
# Shi et al., 2020. Bridging the Gap between Sample-based and
# One-shot Neural Architecture Search with BONAS

import itertools
import os
import random
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from naslib.utils.utils import AverageMeterGroup
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

def accuracy_mse(prediction, target, scale=100.):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)

class MLP(nn.Module):
    def __init__(self, nfeat=72, ifsigmoid=False, layer_size = 64):
        super(MLP, self).__init__()
        self.size = layer_size
        self.ifsigmoid = ifsigmoid
        self.fc1 = nn.Linear(nfeat, self.size)
        self.fc2 = nn.Linear(self.size, self.size)
        self.fc3 = nn.Linear(self.size, self.size)
        self.fc4 = nn.Linear(self.size, self.size)
        self.fc5 = nn.Linear(self.size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight, a=-0.05, b=0.05)
        self.fc1.bias.data.fill_(0)
        nn.init.uniform_(self.fc2.weight, a=-0.05, b=0.05)
        self.fc2.bias.data.fill_(0)
        nn.init.uniform_(self.fc3.weight, a=-0.05, b=0.05)
        self.fc3.bias.data.fill_(0)
        nn.init.xavier_uniform(self.fc4.weight, gain=nn.init.calculate_gain('relu'))
        self.fc4.bias.data.fill_(0)
        self.fc5.bias.data.fill_(0)

    def forward(self, x, extract_embedding=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        embedding = self.fc4(x)
        x = self.fc5(embedding)
        if extract_embedding:
            return embedding
        if self.ifsigmoid:
            return self.sigmoid(x)
        else:
            return x

class BonasMLPPredictor(Predictor):
    def __init__(self, encoding_type='gcn'):
        self.encoding_type = encoding_type

    def get_model(self, **kwargs):
        predictor = MLP(**kwargs)
        return predictor

    def fit(self,xtrain,ytrain, 
            gcn_hidden=144,seed=0,batch_size=7,
            epochs=100,lr=1e-3,wd=0):

        # get mean and std, normlize accuracies
        print('ytrain:')
        print(ytrain)
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean)/self.std
        #ytrain_normed = (np.array(ytrain)) 
        # encode data in gcn format
        train_data = []
        
        for i, arch in enumerate(xtrain):
            encoded = encode(arch, encoding_type=self.encoding_type)
            encoded['val_acc'] = float(ytrain_normed[i])
            train_data.append(encoded)
            
        feat_dim = train_data[0]['feature'].shape[1]
        train_data = np.array(train_data)
        print(train_data[0]['feature'].shape)

        self.model = self.get_model(nfeat=feat_dim)
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for i in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            print('epoch: {}'.format(i))
            for _, batch in enumerate(data_loader):
                feat, target =  batch["feature"], batch["val_acc"].float()
                #print('feature: {}, adjmat: {}'.format(feat,adjmat))
                #print('feature: {},'.format(feat))
                prediction = self.model(batch["feature"])
                loss = criterion(prediction, target)
                #print("prediction: {}, target: {}".format(prediction, target))
                loss.backward()
                optimizer.step()
                mse = accuracy_mse(prediction, target)
                meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))

            lr_scheduler.step()
        train_pred = (self.query(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=1000):
        test_data = np.array([encode(arch,encoding_type=self.encoding_type)
                            for arch in xtest])
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)
        print('test data length: {}'.format(len(test_data)))
        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch["feature"])
                pred.extend(prediction)
        # print('test predictions:')
        # print(pred)
        pred = np.concatenate(pred)
        return pred * self.std + self.mean
