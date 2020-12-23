# Author: Yang Liu @ Abacus.ai
# This is an implementation of the semi-supervised predictor for NAS from the paper:
# Luo et al., 2020. "Semi-Supervised Neural Architecture Search" https://arxiv.org/abs/2002.10389
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from naslib.utils.utils import AverageMeterGroup, AverageMeter
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# default parameters from the paper
n = 1100
m = 10000
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
iteration = 3 #3

use_cuda = True
# decreasing the number of epochs to reduce training time
pretrain_epochs = 200
epochs = 200

nb201_adj_matrix = np.array(
            [[0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float32)

def move_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def convert_arch_to_seq(matrix, ops):
    seq = []
    n = len(matrix)
    max_n = 8
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
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout, bidirectional=True)
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
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
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
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb, new_predict_value

SOS_ID = 0
EOS_ID = 0


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
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        
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
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
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
        )
        self.decoder = Decoder(
            decoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            decoder_length,
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_value, decoder_outputs, archs
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb, new_predict_value = self.encoder.infer(
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

    logging.info('Train data: {}'.format(len(train_input)))
    controller_train_dataset = ControllerDataset(train_input, train_target, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_train(controller_train_queue, model, optimizer)
        if epoch % 10 == 0:
            print("epoch {} train loss {} mse {} ce {}".format(epoch, loss, mse, ce) )

# currently only works for nb201 
def generate_synthetic_controller_data(model, base_arch=None, random_arch=0):
    '''
    base_arch: a list of seq 
    '''
    random_synthetic_input = []
    random_synthetic_target = []
    if random_arch > 0:
        while len(random_synthetic_input) < random_arch:
            ops = [random.randint(1,5) for _ in range(6)]
            ops = [0, *ops, 6]
            seq = convert_arch_to_seq(nb201_adj_matrix, ops)
            if seq not in random_synthetic_input and seq not in base_arch:
                random_synthetic_input.append(seq)

        nao_synthetic_dataset = ControllerDataset(random_synthetic_input, None, False)
        nao_synthetic_queue = torch.utils.data.DataLoader(nao_synthetic_dataset, batch_size=len(nao_synthetic_dataset), shuffle=False, pin_memory=True, drop_last=False)

        with torch.no_grad():
            model.eval()
            for sample in nao_synthetic_queue:
                if use_cuda:
                    encoder_input = move_to_cuda(sample['encoder_input'])
                else:
                    encoder_input = sample['encoder_input']
                _, _, _, predict_value = model.encoder(encoder_input)
                random_synthetic_target += predict_value.data.squeeze().tolist()
        assert len(random_synthetic_input) == len(random_synthetic_target)

    synthetic_input = random_synthetic_input
    synthetic_target = random_synthetic_target
    assert len(synthetic_input) == len(synthetic_target)
    return synthetic_input, synthetic_target

class SemiNASPredictor(Predictor):
    def __init__(self, encoding_type='gcn'):
        self.encoding_type = encoding_type

    def get_model(self, **kwargs):
        predictor = NAO()
        return predictor

    def fit(self,xtrain,ytrain, 
            gcn_hidden=64,seed=0,batch_size=100,
            epochs=1000,lr=1e-3,wd=0):
        up_sample_ratio = 10
        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean)/self.std
        # encode data in seq
        train_seq_pool = []
        train_target_pool = []
        for i, arch in enumerate(xtrain):
            encoded = encode(arch, encoding_type=self.encoding_type)
            seq = convert_arch_to_seq(encoded['adjacency'],encoded['operations'])
            train_seq_pool.append(seq)
            train_target_pool.append(ytrain_normed[i])

        self.model = NAO(
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
        )

        for i in range(iteration+1):
            print('Iteration {}'.format(i+1))

            train_encoder_input = train_seq_pool
            train_encoder_target = train_target_pool

            # Pre-train
            print('Pre-train EPD')
            train_controller(self.model, train_encoder_input, train_encoder_target, pretrain_epochs)
            print('Finish pre-training EPD')
            # Generate synthetic data
            print('Generate synthetic data for EPD')
            synthetic_encoder_input, synthetic_encoder_target = generate_synthetic_controller_data(self.model, train_encoder_input, m)
            if up_sample_ratio is None:
                up_sample_ratio = np.ceil(m / len(train_encoder_input)).astype(np.int)
            else:
                up_sample_ratio = up_sample_ratio

            all_encoder_input = train_encoder_input * up_sample_ratio + synthetic_encoder_input
            all_encoder_target = train_encoder_target * up_sample_ratio + synthetic_encoder_target
            # Train
            print('Train EPD')
            train_controller(self.model, all_encoder_input, all_encoder_target, epochs)
            print('Finish training EPD')
            
        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=100):

        test_seq_pool = []
        for i, arch in enumerate(xtest):
            encoded = encode(arch, encoding_type=self.encoding_type)
            seq = convert_arch_to_seq(encoded['adjacency'],encoded['operations'])
            test_seq_pool.append(seq)

        test_dataset = ControllerDataset(test_seq_pool, None, False)
        test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False) 

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, sample in enumerate(test_queue):
                #print(sample)
                encoder_input = move_to_cuda(sample['encoder_input'])
                decoder_target = move_to_cuda(sample['decoder_target'])
                prediction, _, _ = self.model(encoder_input, decoder_target)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return pred * self.std + self.mean
