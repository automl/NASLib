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


class LSTM(nn.Module):
    def __init__(self, nfeat, timestep):
        self.emb_dim = 100
        self.hidden_dim = 100
        self.timestep = timestep
        super(LSTM, self).__init__()
        self.adj_emb = nn.Embedding(2, embedding_dim=self.emb_dim)
        nn.init.uniform_(self.adj_emb.weight, a=-0.1, b=0.1)
        self.op_emb = nn.Embedding(nfeat, embedding_dim=self.emb_dim)
        nn.init.uniform_(self.op_emb.weight, a=-0.1, b=0.1)
        self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * self.timestep, 1)

    def forward(self, x, adj):
        op = x
        adj_embed = self.adj_emb(adj)
        op_embed = self.op_emb(op)
        embed = torch.cat((adj_embed, op_embed), 1)
        out, (h_n, c_n) = self.rnn(embed)
        out = out.contiguous().view(-1, out.shape[1] * out.shape[2])
        out = self.fc(out)
        return out


class BonasLSTMPredictor(Predictor):
    def __init__(self, encoding_type='gcn'):
        self.encoding_type = encoding_type

    def get_model(self, **kwargs):
        predictor = LSTM()
        return predictor

    def fit(self,xtrain,ytrain, 
            gcn_hidden=144,seed=0,batch_size=7,
            epochs=100,lr=1e-3,wd=0):

        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean)/self.std
        # encode data in gcn format
        train_data = []
        for i, arch in enumerate(xtrain):
            encoded = encode(arch, encoding_type=self.encoding_type)
            encoded['val_acc'] = float(ytrain_normed[i])
            train_data.append(encoded)
        train_data = np.array(train_data)

        self.model = self.get_model(gcn_hidden=gcn_hidden)
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for _ in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            for _, batch in enumerate(data_loader):
                feat, adjmat, target =  batch["operations"], batch["adjacency"], batch["val_acc"].float()
                arch = np.concatenate(adjmat,feat)
                prediction = self.model(arch)
                loss = criterion(prediction, target)
                print("prediction: {}, target: {}".format(prediction, target))
                loss.backward()
                optimizer.step()
                mse = accuracy_mse(prediction, target)
                meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))

            lr_scheduler.step()
        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=1000):
        test_data = np.array([encode(arch,encoding_type=self.encoding_type)
                            for arch in xtest])
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                feat, adjmat =  batch["operations"], batch["adjacency"]
                arch = np.concatenate(adjmat,feat)
                prediction = self.model(arch)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return pred * self.std + self.mean
