# Author: Yang Liu @ Abacus.ai
# This is an implementation of the semi-supervised predictor for NAS from the paper:
# Luo et al., 2020. "Semi-Supervised Neural Architecture Search" https://arxiv.org/abs/2002.10389
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import copy
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

from naslib.utils import utils
from naslib.utils.utils import AverageMeterGroup, AverageMeter

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform
from naslib.predictors.predictor import Predictor
from naslib.predictors.lcsvr import loguniform
from naslib.predictors.zerocost_v1 import ZeroCostV1

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_naslib
from naslib.search_spaces import NasBench201SearchSpace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# default parameters from the paper
n = 1100
# m = 10000
nodes = 8
new_arch = 300
k = 100
encoder_layers = 1
hidden_size = 64
mlp_layers = 2
mlp_hidden_size = 16
decoder_layers = 1
source_length =35 #27
encoder_length = 35 #27
decoder_length = 35 #27
dropout = 0.1 
l2_reg = 1e-4
vocab_size = 9 # 7 original
max_step_size = 100
trade_off = 0.8  
up_sample_ratio = 10
batch_size = 100 
lr = 0.001
optimizer = 'adam'
grad_bound = 5.0

# helper to move object to cuda when available
def move_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# conver adjacenty matrix + ops encoding to seminas sequence
def convert_arch_to_seq(matrix, ops, max_n=8):
    seq = []
    n = len(matrix)
    max_n = max_n
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            seq += [0 for i in range(col)]
            seq.append(0)
        else:
            for row in range(col):
                seq.append(matrix[row][col]+1)
            seq.append(ops[col]+2)

    assert len(seq) == (max_n+2)*(max_n-1)/2
    return seq

# convert seminas sequence back to adjacenty matrix + ops encoding
def convert_seq_to_arch(seq):
    n = int(math.floor(math.sqrt((len(seq) + 1) * 2)))
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    ops = [0]
    for i in range(n-1):
        offset=(i+3)*i//2
        for j in range(i+1):
            matrix[j][i+1] = seq[offset+j] - 1
        ops.append(seq[offset+i+1]-2)
    return matrix, ops

# NAO dataset
class ControllerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
        super(ControllerDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]
        if self.train:
            decoder_input = [self.sos_id] + encoder_input[:-1]
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'encoder_target': torch.FloatTensor(encoder_target),
                'decoder_input': torch.LongTensor(decoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
        else:
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
            if encoder_target is not None:
                sample['encoder_target'] = torch.FloatTensor(encoder_target)
        return sample
    
    def __len__(self):
        return len(self.inputs)

class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 mlp_layers,
                 hidden_size,
                 mlp_hidden_size,
                 vocab_size,
                 dropout,
                 source_length,
                 length,
                 ):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mlp = nn.ModuleList([])
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.append(nn.Linear(self.hidden_size, self.mlp_hidden_size))
            elif i == self.mlp_layers - 1:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.hidden_size))
            else:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size))
        self.regressor = nn.Linear(self.hidden_size, 1)
    
    def forward_predictor(self, x):
        residual = x
        for i, mlp_layer in enumerate(self.mlp):
            x = mlp_layer(x)
            x = F.relu(x)
            if i != self.mlp_layers:
                x = F.dropout(x, self.dropout, training=self.training)
        x = (residual + x) * math.sqrt(0.5)
        x = self.regressor(x)
        predict_value = x
        return predict_value

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        residual = x
        x, hidden = self.rnn(x)
        x = self.out_proj(x)
        x = residual + x
        x = F.normalize(x, 2, dim=-1)
        encoder_outputs = x
        encoder_hidden = hidden
        
        x = torch.mean(x, dim=1)
        x = F.normalize(x, 2, dim=-1)
        arch_emb = x
        
        residual = x
        for i, mlp_layer in enumerate(self.mlp):
            x = mlp_layer(x)
            x = F.relu(x)
            if i != self.mlp_layers:
                x = F.dropout(x, self.dropout, training=self.training)
        x = (residual + x) * math.sqrt(0.5)
        x = self.regressor(x)
        predict_value = x
        return encoder_outputs, encoder_hidden, arch_emb, predict_value
    
    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, 
                                               torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        new_predict_value = self.forward_predictor(new_arch_emb)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, \
    new_arch_emb, new_predict_value

SOS_ID = 0
EOS_ID = 0

# attention module
class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
    
    def forward(self, input, source_hids, mask=None):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        
        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)
        
        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim)))\
        .view(batch_size, -1, self.output_dim)
        
        return output, attn

class Decoder(nn.Module):
    
    def __init__(self,
                 layers,
                 hidden_size,
                 vocab_size,
                 dropout,
                 length,
                 ):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, 
                           batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.n = int(math.floor(math.sqrt((self.length + 1) * 2)))
        self.offsets=[]
        for i in range(self.n):
            self.offsets.append( (i + 3) * i // 2 - 1)
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_hidden = self._init_state(encoder_hidden)
        if x is not None:
            bsz = x.size(0)
            tgt_len = x.size(1)
            x = self.embedding(x)
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            predicted_softmax = F.log_softmax(self.out(x.view(-1, self.hidden_size)), dim=-1)
            predicted_softmax = predicted_softmax.view(bsz, tgt_len, -1)
            return predicted_softmax, None

        # inference
        assert x is None
        bsz = encoder_hidden[0].size(1)
        length = self.length
        decoder_input = encoder_hidden[0].new(bsz, 1).fill_(0).long()
        decoded_ids = encoder_hidden[0].new(bsz, 0).fill_(0).long()
        
        def decode(step, output):
            if step in self.offsets:  # sample operation, should be in [3, 7]
                symbol = output[:, 3:].topk(1)[1] + 3
            else:  # sample connection, should be in [1, 2]
                symbol = output[:, 1:3].topk(1)[1] + 1
            return symbol
        
        for i in range(length):
            x = self.embedding(decoder_input[:, i:i+1])
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, decoder_hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            output = self.out(x.squeeze(1))
            symbol = decode(i, output)
            decoded_ids = torch.cat((decoded_ids, symbol), axis=-1)
            decoder_input = torch.cat((decoder_input, symbol), axis=-1)

        return None, decoded_ids
    
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

# NAO predictor with Encoder and Decoder
class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 mlp_layers,
                 hidden_size,
                 mlp_hidden_size,
                 vocab_size,
                 dropout,
                 source_length,
                 encoder_length,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            mlp_layers,
            hidden_size,
            mlp_hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length,
        ).to(device)
        self.decoder = Decoder(
            decoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            decoder_length,
        ).to(device)

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable.to(device))
        decoder_hidden = (arch_emb.unsqueeze(0).to(device),
                          arch_emb.unsqueeze(0).to(device))
        decoder_outputs, archs = self.decoder(target_variable.to(device),
                                              decoder_hidden,
                                              encoder_outputs.to(device))
        return predict_value, decoder_outputs, archs
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, \
        new_encoder_outputs, new_arch_emb, new_predict_value = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs, new_predict_value

def controller_train(train_queue, model, optimizer):

    objs = AverageMeter()
    mse = AverageMeter()
    nll = AverageMeter()
    model.train()
    for step, sample in enumerate(train_queue):

        encoder_input = move_to_cuda(sample['encoder_input'])
        encoder_target = move_to_cuda(sample['encoder_target'])
        decoder_input = move_to_cuda(sample['decoder_input'])
        decoder_target = move_to_cuda(sample['decoder_target'])

        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = trade_off * loss_1 + (1 - trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
    
    return objs.avg, mse.avg, nll.avg


def controller_infer(queue, model, step, direction='+'):
    new_arch_list = []
    new_predict_values = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = move_to_cuda(sample['encoder_input'])
        model.zero_grad()
        new_arch, new_predict_value = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
        new_predict_values.extend(new_predict_value.data.squeeze().tolist())
    return new_arch_list, new_predict_values


def train_controller(model, train_input, train_target, epochs):

    controller_train_dataset = ControllerDataset(train_input, train_target, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_train(controller_train_queue, model, optimizer)
        if epoch % 10 == 0:
            print("epoch {} train loss {} mse {} ce {}".format(epoch, loss, mse, ce) )


def discretize(x, upper_bounds=None, one_hot=False):
    # return discretization based on upper_bounds
    # supports one_hot or categorical output
    assert upper_bounds is not None and len(upper_bounds) >= 1

    if one_hot:
        cat = len(upper_bounds) + 1
        discretized = [0 for _ in range(cat)]
        for i, ub in enumerate(upper_bounds):
            if x < ub:
                discretized[i] = 1
                return discretized
        discretized[-1] =  1
        return discretized
    else:
        for i, ub in enumerate(upper_bounds):
            if x < ub:
                return i
        return len(upper_bounds) + 1

def get_bins(zero_cost, train_size):
    """
    The SemiNAS predictor uses a discrete encoding, so we must discretize the (continuous) 
    zero-cost features. We do this by putting them into bins. In a real experiment, we 
    would need to estimate the upper bounds for each bin during the search. To save time, 
    we precomputed the bins and then add the runtime of this precomputation later.
    """
    if zero_cost == 'jacov':
        if train_size < 10:
            # precomputation based on 100 jacov values (366 seconds)
            bins = [-19838.279, -906.12, -444.588, -366.404, -316.694, 
                    -285.499, -283.021, -280.614, -278.303]
        else:
            # precomputed based on 1000 jacov values (3660 seconds)
            bins = [-20893.873, -1179.832, -518.407, -373.523, -317.264, 
                    -284.944, -281.242, -279.503, -278.083]
    else:
        raise NotImplementedError('Currently no other zero-cost methods are supported')
    
    return bins


class OmniSemiNASPredictor(Predictor):
    # todo: make the code general to support any zerocost predictors
    def __init__(self, encoding_type='seminas', ss_type=None, semi=True, hpo_wrapper=False, 
                 config=None, run_pre_compute=True, jacov_onehot=True, synthetic_factor=1, 
                 max_zerocost=np.inf, zero_cost=['jacov'], lce=[]):
        self.encoding_type = encoding_type
        self.semi = semi
        self.synthetic_factor = synthetic_factor
        self.lce = lce
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.max_zerocost = max_zerocost
        self.default_hyperparams = {'gcn_hidden':64, 
                                    'batch_size':100, 
                                    'lr':1e-3}
        self.hyperparams = None
        self.config = config
        self.zero_cost = zero_cost
        self.run_pre_compute = run_pre_compute
        self.jacov_onehot = jacov_onehot

        self.jacov_bins = get_bins('jacov', 100) # todo: this should go into fit()
        self.bins = len(self.jacov_bins) + 1

        # set additional feature length and vocabulary size based on encoding type and number of bins
        self.jacov_vocab = 2 if self.jacov_onehot else self.bins + 1
        self.jacov_length =  self.bins if self.jacov_onehot else 1

    def prepare_features(self, xdata, info=None):
        # this concatenates architecture features with zero-cost features        
        full_xdata = [[] for _ in range(len(xdata))]
                
        if self.encoding_type is not None:
            # convert the architecture to a categorical encoding
            for i, arch in enumerate(xdata):
                encoded = encode(arch, encoding_type=self.encoding_type, 
                                 ss_type=self.ss_type)
                seq = convert_arch_to_seq(encoded['adjacency'], 
                                          encoded['operations'], 
                                          max_n=self.max_n)
                full_xdata[i] = [*full_xdata[i], *seq]

        if len(self.zero_cost) > 0 and self.train_size <= self.max_zerocost: 
            # add zero_cost features here
            for key in self.zero_cost:
                for i, arch in enumerate(xdata):
                    # todo: the following code is still specific to jacov. Make it for any zerocost
                    if self.jacov_onehot:
                        jac_encoded = discretize(info['jacov_scores'][i], upper_bounds=self.jacov_bins, 
                                                    one_hot=self.jacov_onehot) 
                        jac_encoded = [jac + self.jacov_offset for jac in jac_encoded]
                    else:
                        jac_encoded = discretize(info['jacov_scores'][i], upper_bounds=self.jacov_bins, 
                                                 one_hot=self.jacov_onehot) + self.jacov_offset

                    full_xdata[i] = [*full_xdata[i], *jac_encoded]

        if 'sotle' in self.lce and len(info[0]['TRAIN_LOSS_lc']) >= 3:
            train_losses = np.array([lcs['TRAIN_LOSS_lc'][-1] for lcs in info])
            # todo: discretize train_losses
            full_xdata = [[*x, train_losses[i]] for i, x in enumerate(full_xdata)]

        elif 'sotle' in self.lce and len(info[0]['TRAIN_LOSS_lc']) < 3:
            logger.info('Not enough fidelities to use train loss')

        return full_xdata

    def generate_synthetic_labels(self, model, synthetic_input):

        # use the model to label the synthetic data
        synthetic_dataset = ControllerDataset(synthetic_input, None, False)
        synthetic_queue = torch.utils.data.DataLoader(synthetic_dataset, batch_size=len(synthetic_dataset), 
                                                      shuffle=False, pin_memory=True, drop_last=False)

        synthetic_target = []
        with torch.no_grad():
            model.eval()
            for sample in synthetic_queue:
                encoder_input = move_to_cuda(sample['encoder_input'])
                _, _, _, predict_value = model.encoder(encoder_input)
                synthetic_target += predict_value.data.squeeze().tolist()
        assert len(synthetic_input) == len(synthetic_target)
        return synthetic_target

    def fit(self, xtrain, ytrain, train_info=None,
            wd=0, iterations=1, epochs=50,
            pretrain_epochs=50):
        
        self.train_size = len(xtrain)
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        batch_size = self.hyperparams['batch_size']
        gcn_hidden = self.hyperparams['gcn_hidden']
        lr = self.hyperparams['lr']
        up_sample_ratio = 10

        # todo: these could be made non class attributes
        if self.ss_type == 'nasbench101':
            self.max_n = 7
            self.encoder_length=27
            self.decoder_length=27
            self.vocab_size=7
        elif self.ss_type == 'nasbench201':
            self.max_n = 8
            self.encoder_length=35
            self.decoder_length=35
            self.vocab_size=9 
        elif self.ss_type == 'darts':
            self.max_n = 35 
            self.encoder_length=629
            self.decoder_length=629
            self.vocab_size=13

        self.jacov_offset = copy.copy(self.vocab_size)
        self.encoder_length += self.jacov_length
        self.decoder_length += self.jacov_length
        self.vocab_size += self.jacov_vocab

        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean) / self.std
        
        self.model = NAO(encoder_layers,
                         decoder_layers,
                         mlp_layers,
                         hidden_size,
                         mlp_hidden_size,
                         self.vocab_size,
                         dropout,
                         source_length,
                         self.encoder_length,
                         self.decoder_length
                         ).to(device)

        xtrain_full_features = self.prepare_features(xtrain, info=self.xtrain_zc_info)

        for i in range(iterations):
            print('Iteration {}'.format(i+1))

            # Pre-train
            print('Pre-train EPD')
            train_controller(self.model, xtrain_full_features, ytrain_normed, pretrain_epochs)
            print('Finish pre-training EPD')

            if self.semi:
                # Check that we have unlabeled data from either pre_compute() or set_pre_compute()
                assert self.unlabeled is not None, 'Unlabeled data was never generated'
                print('Generate synthetic data for EPD')
                num_synthetic = self.synthetic_factor * len(xtrain)
                synthetic_full_features = self.prepare_features(self.unlabeled, self.unlabeled_zc_info)
                synthetic_full_features = synthetic_full_features[:num_synthetic]
                synthetic_target = self.generate_synthetic_labels(self.model, synthetic_full_features)
                if up_sample_ratio is None:
                    up_sample_ratio = np.ceil(m / len(xtrain_full_features)).astype(np.int)
                else:
                    up_sample_ratio = up_sample_ratio
                    
                combined_input = xtrain_full_features * up_sample_ratio + synthetic_full_features
                combined_target = list(ytrain_normed) * up_sample_ratio + synthetic_target
                print('Train EPD')
                train_controller(self.model, combined_input, combined_target, epochs)
                print('Finish training EPD')

    def query(self, xtest, info=None, batch_size=100):

        if self.run_pre_compute:
            # if we ran pre_compute(), the xtest zc scores are in self.xtest_zc_info
            test_data = self.prepare_features(xtest, info=self.xtest_zc_info)
        else:
            # otherwise, they will be in info (often used during NAS experiments)
            test_data = self.prepare_features(xtest, info=info)            
        test_dataset = ControllerDataset(test_data, None, False)
        test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                                 pin_memory=True, drop_last=False) 

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, sample in enumerate(test_queue):
                encoder_input = move_to_cuda(sample['encoder_input'])
                decoder_target = move_to_cuda(sample['decoder_target'])
                prediction, _, _ = self.model(encoder_input, decoder_target)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred * self.std + self.mean)

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                'gcn_hidden': int(loguniform(16, 128)), 
                'batch_size': int(loguniform(32, 256)), 
                'lr': loguniform(.00001, .1)}

        self.hyperparams = params
        return params
    
    def pre_compute(self, xtrain, xtest=None, unlabeled=None):
        """
        All of this computation could go into fit() and query(), but we do it
        here to save time, so that we don't have to re-compute Jacobian covariances
        for all train_sizes when running experiment_types that vary train size or fidelity.        
        
        This method computes zerocost info for the train set, test set, and synthetic set
        (if applicable). It also stores the synthetic architectures.
        """
        self.xtrain_zc_info = {}
        self.xtest_zc_info = {}
        self.unlabeled_zc_info = {}
        self.unlabeled = unlabeled

        if len(self.zero_cost) > 0:
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

            for method_name in self.zero_cost:
                # todo: allow ZeroCostV2 as well
                zc_method = ZeroCostV1(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                
                # save the raw scores, since bucketing depends on the train set size
                self.xtrain_zc_info[f'{method_name}_scores'] = zc_method.query(xtrain)
                self.xtest_zc_info[f'{method_name}_scores'] = zc_method.query(xtest)
                if unlabeled is not None:
                    self.unlabeled_zc_info[f'{method_name}_scores'] = zc_method.query(unlabeled)
        
    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None, 
                             xtest_zc_info=None, unlabeled_zc_info=None):
        """
        This is another way to add pre-computations, if they are done outside this class.
        This is currently used in the NAS experiments; since the predictor is retrained constantly,
        it avoids re-computing zero-cost scores.
        """
        if unlabeled is not None:
            self.unlabeled = unlabeled
        if xtrain_zc_info is not None:
            self.xtrain_zc_info = xtrain_zc_info
        if xtest_zc_info is not None:
            self.xtest_zc_info = xtest_zc_info
        if unlabeled_zc_info is not None:
            self.unlabeled_zc_info = unlabeled_zc_info            

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query, such as a partial learning curve,
        or hyperparameters of the architecture
        """
        if len(self.lce) > 0:
            # add the metrics needed for the lce predictors
            required_metric_dict = {'sotle':Metric.TRAIN_LOSS, 'valacc':Metric.VAL_ACCURACY}
            self.metric = [required_metric_dict[key] for key in self.lce]

            reqs = {'requires_partial_lc':True, 
                    'metric':self.metric, 
                    'requires_hyperparameters':False, 
                    'hyperparams':{}, 
                    'unlabeled':self.semi, 
                    'unlabeled_factor':self.synthetic_factor
                   }
        else:
            reqs = {'requires_partial_lc':False, 
                    'metric':None, 
                    'requires_hyperparameters':False, 
                    'hyperparams':{}, 
                    'unlabeled':self.semi, 
                    'unlabeled_factor':self.synthetic_factor}
        return reqs