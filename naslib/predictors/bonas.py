# Author: Yang Liu @ Abacus.ai
# This is an implementation of gcn predictor for NAS from the paper:
# Shi et al., 2020. Bridging the Gap between Sample-based and
# One-shot Neural Architecture Search with BONAS
# We added cosine annealing.

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

from naslib.utils import AverageMeterGroup
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    mx = np.dot(r_mat_inv, mx)
    return mx


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


def add_global_node(mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if ifAdj:
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx


def padzero(mx, ifAdj, maxsize=7):
    if ifAdj:
        while mx.shape[0] < maxsize:
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    else:
        while mx.shape[0] < maxsize:
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    return mx


def net_decoder(operations):
    operations = np.array(operations, dtype=np.int32)
    for i in range(len(operations)):
        if operations[i] == 2:  # input
            operations[i] = 3
        elif operations[i] == 3:  # conv1
            operations[i] = 0
        elif operations[i] == 4:  # pool
            operations[i] = 2
        elif operations[i] == 5:  # conv3
            operations[i] = 1
        elif operations[i] == 6:  # output
            operations[i] = 4
    one_hot = np.zeros((len(operations), 5))
    one_hot[np.arange(len(operations)), operations] = 1
    return one_hot


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj):
        support = torch.matmul(input_, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# nfeat=7 for nasbench 201
class GCN(nn.Module):
    def __init__(self, nfeat=7, ifsigmoid=False, gcn_hidden=64):
        super(GCN, self).__init__()
        self.ifsigmoid = ifsigmoid
        self.size = gcn_hidden
        self.gc1 = GraphConvolution(nfeat, self.size)
        self.gc2 = GraphConvolution(self.size, self.size)
        self.gc3 = GraphConvolution(self.size, self.size)
        self.gc4 = GraphConvolution(self.size, self.size)
        self.bn1 = nn.BatchNorm1d(self.size)
        self.bn2 = nn.BatchNorm1d(self.size)
        self.bn3 = nn.BatchNorm1d(self.size)
        self.bn4 = nn.BatchNorm1d(self.size)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, feat, adj, extract_embedding=False):
        feat = feat.to(device)
        adj = adj.to(device)
        x = F.relu(self.bn1(self.gc1(feat, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(self.gc2(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn3(self.gc3(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn4(self.gc4(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        embeddings = x[:, x.size()[1] - 1, :]
        x = self.fc(embeddings).view(-1)
        if extract_embedding:
            return embeddings
        if self.ifsigmoid:
            return self.sigmoid(x)
        else:
            return x


class BonasPredictor(Predictor):
    def __init__(self, encoding_type="bonas", ss_type=None, hpo_wrapper=False):
        self.encoding_type = encoding_type
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {"gcn_hidden": 64, "batch_size": 128, "lr": 1e-4}
        self.hyperparams = None

    def get_model(self, **kwargs):
        predictor = GCN(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain, train_info=None, epochs=100, wd=0):

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        batch_size = self.hyperparams["batch_size"]
        gcn_hidden = self.hyperparams["gcn_hidden"]
        lr = self.hyperparams["lr"]

        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean) / self.std
        # encode data in gcn format
        train_data = []
        for i, arch in enumerate(xtrain):
            encoded = arch.encode(encoding_type=self.encoding_type)
            encoded["val_acc"] = float(ytrain_normed[i])
            train_data.append(encoded)
        train_data = np.array(train_data)
        nfeat = len(train_data[0]["operations"][0])
        self.model = self.get_model(gcn_hidden=gcn_hidden, nfeat=nfeat)
        data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.model.to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0
        )

        self.model.train()

        for _ in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            for _, batch in enumerate(data_loader):
                feat, adjmat, target = (
                    batch["operations"].to(device),
                    batch["adjacency"].to(device),
                    batch["val_acc"].float().to(device),
                )
                prediction = self.model(feat, adjmat)
                # print('predictions:\n{}'.format(prediction))
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                mse = accuracy_mse(prediction, target)
                meters.update(
                    {"loss": loss.item(), "mse": mse.item()}, n=target.size(0)
                )

            lr_scheduler.step()
        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=100):
        test_data = np.array(
            [
                arch.encode(encoding_type=self.encoding_type)
                for arch in xtest
            ]
        )
        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, drop_last=False
        )

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                feat, adjmat = batch["operations"], batch["adjacency"]
                prediction = self.model(feat, adjmat)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return pred * self.std + self.mean

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                "gcn_hidden": int(loguniform(16, 128)),
                "batch_size": int(loguniform(32, 256)),
                "lr": loguniform(0.00001, 0.1),
            }

        self.hyperparams = params
        return params
