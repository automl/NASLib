# Author: Yang Liu @ Abacus.ai
# This is an implementation of the semi-supervised predictor for NAS from the paper:
# Luo et al., 2020. "Semi-Supervised Neural Architecture Search" https://arxiv.org/abs/2002.10389
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from naslib.utils.utils import AverageMeterGroup, AverageMeter
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform


from naslib.predictors.predictor import Predictor
from naslib.predictors.lcsvr import loguniform
from naslib.predictors.zerocost_estimators import ZeroCostEstimators
#from naslib.predictors.utils.encodings import encode
from naslib.utils import utils
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_naslib
from naslib.search_spaces import NasBench201SearchSpace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# iteration = 3

use_cuda = True
# pretrain_epochs = 1000
# epochs = 1000

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
        # print("x shape: \n{}".format(x.shape))
        # print('x max: \n {}'.format(torch.max(torch.tensor(x))))
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

# for nb 101
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9

def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])
    
    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes
#for nb 101


def num_edges_and_vertices(matrix):
    # return the true number of edges and vertices
    edges, nodes = get_utilized(matrix)
    return len(edges), len(nodes) 

def sample_random_architecture_nb101():
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice([2,3,4], size=NUM_VERTICES).tolist()
            ops[0] = 0 #INPUT
            ops[-1] = 1 #OUTPUT
            num_edges, num_vertices = num_edges_and_vertices(matrix)
            if num_edges > 1 and num_edges < 10:
                break      
        return {'matrix':matrix, 'ops':ops}

def encode_darts(arch):
    matrices = []
    ops = []
    for cell in arch:
        mat,op = transform_matrix(cell)
        matrices.append(mat)
        ops.append(op)

    matrices[0] = add_global_node(matrices[0],True)
    matrices[1] = add_global_node(matrices[1],True)
    matrices[0] = np.transpose(matrices[0])
    matrices[1] = np.transpose(matrices[1])
    
    ops[0] = add_global_node(ops[0],False)
    ops[1] = add_global_node(ops[1],False)

    mat_length = len(matrices[0][0])
    merged_length = len(matrices[0][0])*2
    matrix_final = np.zeros((merged_length,merged_length))

    for col in range(mat_length):
        for row in range(col):
            matrix_final[row,col] = matrices[0][row,col]
            matrix_final[row+mat_length,col+mat_length] = matrices[1][row,col]

    ops_onehot = np.concatenate((ops[0],ops[1]),axis=0)

    matrix_final = add_global_node(matrix_final,True)
    ops_onehot = add_global_node(ops_onehot,False)
    
    matrix_final = np.array(matrix_final,dtype=np.float32)
    ops_onehot = np.array(ops_onehot,dtype=np.float32)
    ops = [np.where(r==1)[0][0] for r in ops_onehot]

    dic = {
        'adjacency': matrix_final,
        'operations': ops,
        'val_acc': 0.0
    }
    return dic

def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx

def transform_matrix(cell):
    normal = cell

    node_num = len(normal)+3

    adj = np.zeros((node_num, node_num))

    ops = np.zeros((node_num, 8)) # 6+2 operations 
    for i in range(len(normal)):
        connect, op = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+2] = 1
        else:
            adj[(connect-2)*2+2][i+2] = 1
            adj[(connect-2)*2+3][i+2] = 1
        ops[i+2][op] = 1
    adj[2:-1, -1] = 1
    ops[0:2, 0] = 1
    ops[-1][-1] = 1
    return adj, ops

def generate_arch(ss_type,info=None):
    ops = []
    if ss_type == 'nasbench101':
        spec = sample_random_architecture_nb101()
        seq = convert_arch_to_seq(spec['matrix'],spec['ops'],max_n=7)
    elif ss_type == 'nasbench201':
        ops = [random.randint(1,5) for _ in range(6)]
        ops_padded = [0, *ops, 6]
        ops = [op - 1  for op in ops]  #zero index for later conversion
        seq = convert_arch_to_seq(nb201_adj_matrix, ops_padded)
    elif ss_type == 'darts':
        cell_norm = [( random.randint(0,i//2+1), random.randint(0,6) ) for i in range(8)]
        cell_reduct = [( random.randint(0,i//2+1), random.randint(0,6) ) for i in range(8)]
        cells = [cell_norm, cell_reduct]
        arch = encode_darts(cells)
        seq = convert_arch_to_seq(arch['adjacency'],arch['operations'],max_n=35)

    return seq,ops


def discretize(x, upper_bounds=[-3,-2,-1,0,1,2,3], one_hot=False):
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

class SemiNASJCPredictor(Predictor):
    def __init__(self, encoding_type='seminas', ss_type=None, semi=False, hpo_wrapper=False, 
                 config=None, run_pre_compute=True, jacov_onehot=True):
        self.encoding_type = encoding_type
        self.semi = True #semi
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {'gcn_hidden':64, 
                                    'batch_size':100, 
                                    'lr':1e-3}
        self.hyperparams = None

        self.config = config
        self.zero_cost = ['jacov']
        self.run_pre_compute = run_pre_compute
        self.jacov_onehot = jacov_onehot # one_hot encoding works better
        print('jacov onehot encoding: {}'.format(self.jacov_onehot))
        self.jacov_bins = [-6479.906262535439, -1048.4814023716435, -478.08807967011205,
                           -354.1177984864107, -302.3988674730198, -283.3622277685421, 
                           -280.84198418968407, -279.2643230054042, -277.87210246240795]

        if self.jacov_bins is None:
            self.bins = 10 # use 10 bins by default
        else:
            self.bins = len(self.jacov_bins) + 1
        print(self.bins)
        self.jacov_vocab = 2 if self.jacov_onehot else self.bins + 1
        self.jacov_length =  self.bins if self.jacov_onehot else 1
        self.jacov_train_mean = None
        self.jacov_train_std = None

    def pre_compute(self, xtrain, xtest):
        """
        All of this computation could go into fit() and query(), but we do it
        here to save time, so that we don't have to re-compute Jacobian covariances
        for all train_sizes when running experiment_types that vary train size or fidelity.        
        """
        self.xtrain_zc_info = {}
        self.xtest_zc_info = {}

        if len(self.zero_cost) > 0:
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

            for method_name in self.zero_cost:
                #print('pre computing')
                zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                xtrain_zc_scores = zc_method.query(xtrain)
                xtest_zc_scores = zc_method.query(xtest)
                xtrain_zc_scores_raw = copy.copy(xtrain_zc_scores)
                upper_bounds = []
                for i in range(1,10):
                    upper_bounds.append(np.quantile(xtrain_zc_scores_raw,i/10.0))
                print('estimated upper bounds:')
                print(upper_bounds)
                if self.jacov_bins is None:
                    self.jacov_bins = upper_bounds

                if self.jacov_train_mean is None:
                    self.jacov_train_mean = np.mean(np.array(xtrain_zc_scores)) 
                    self.jacov_train_std = np.std((np.array(xtrain_zc_scores)))
                
                #normalized_train = (np.array(xtrain_zc_scores) - self.jacov_train_mean)/self.jacov_train_std
                #normalized_test = (np.array(xtest_zc_scores) - self.jacov_train_mean)/self.jacov_train_std
                
                self.xtrain_zc_info[f'{method_name}_scores'] = xtrain_zc_scores #normalized_train
                self.xtest_zc_info[f'{method_name}_scores'] = xtest_zc_scores #normalized_test

    def prepare_features(self, xdata, info, train=True):
        # prepare training data features
        full_xdata = [[] for _ in range(len(xdata))]
        if len(self.zero_cost) > 0: # and self.train_size <= self.max_zerocost: 
            if self.run_pre_compute:
                for key in self.xtrain_zc_info:
                    if train:
                        full_xdata = [[*x, self.xtrain_zc_info[key][i]] for i, x in enumerate(full_xdata)]
                    else:
                        full_xdata = [[*x, self.xtest_zc_info[key][i]] for i, x in enumerate(full_xdata)]
            else:
                # if the zero_cost scores were not precomputed, they are in info
                full_xdata = [[*x, info[i]] for i, x in enumerate(full_xdata)]
            #print(full_xdata)
        
        return np.array(full_xdata)

    def get_model(self, **kwargs):
        # old API, not being used 
        if self.ss_type == 'nasbench101':
            predictor = NAO(encoder_length=27,decoder_length=27)
        elif self.ss_type == 'nasbench201':
            predictor = NAO(encoder_length=35,decoder_length=35)
        elif self.ss_type == 'darts':
            predictor = NAO(encoder_length=629,decoder_length=629,vocab_size=12)
        return predictor

    # currently only works for nb201 
    def generate_synthetic_controller_data(self, model, base_arch=None, random_arch=0,ss_type=None):
        '''
        base_arch: a list of seq 
        '''
        ops = []
        random_synthetic_input = []
        random_synthetic_target = []
        if random_arch > 0:
            while len(random_synthetic_input) < random_arch:
                seq,op = generate_arch(ss_type=ss_type)
                if seq not in random_synthetic_input and seq not in base_arch:
                    random_synthetic_input.append(seq)
                    ops.append(op)

            naslib_object = NasBench201SearchSpace()
            archs = []
            for i, op in enumerate(ops):
                arch = copy.deepcopy(naslib_object)
                convert_op_indices_to_naslib(op,arch)
                archs.append(arch)
            jacovs = self.prepare_features(archs, self.train_info, train=True)
            random_synthetic_input_no_jc = copy.deepcopy(random_synthetic_input)
            random_synthetic_input = []
            for i, seq in enumerate(random_synthetic_input_no_jc):
                #print("jacov:{}".format(jacovs[i]))
                if self.jacov_onehot:
                    jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins,one_hot=self.jacov_onehot) 
                    jac_encoded = [jac +self.jacov_offset for jac in jac_encoded]
                    seq.extend(jac_encoded)
                else:
                    jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins, one_hot=self.jacov_onehot) + self.jacov_offset
                    seq.append(jac_encoded)
                #print("synthetic seq")
                #print(seq)
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

    def fit(self, xtrain, ytrain, train_info=None,
            wd=0, iteration=1, epochs=50,
            pretrain_epochs=50, 
            synthetic_factor=1):
        
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        self.train_info = train_info

        batch_size = self.hyperparams['batch_size']
        gcn_hidden = self.hyperparams['gcn_hidden']
        lr = self.hyperparams['lr']

        up_sample_ratio = 10

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
        print(self.vocab_size)
        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean)/self.std
        # encode data in seq
        train_seq_pool = []
        train_target_pool = []
        jacovs = self.prepare_features(xtrain, self.train_info, train=True)
        #print('original training set:')
        for i, arch in enumerate(xtrain):
            #print("architecture")
            #print(arch)
            encoded = encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
            seq = convert_arch_to_seq(encoded['adjacency'],encoded['operations'],max_n=self.max_n)
            #print(seq)
            #print(len(seq.append(jacovs[i])))
            if self.jacov_onehot:
                jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins,one_hot=self.jacov_onehot) 
                jac_encoded = [jac +self.jacov_offset for jac in jac_encoded]
                seq.extend(jac_encoded)
            else:
                jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins,one_hot=self.jacov_onehot) + self.jacov_offset
                seq.append(jac_encoded)
            #print(seq)
            train_seq_pool.append(seq)
            train_target_pool.append(ytrain_normed[i])

        self.model = NAO(
            encoder_layers,
            decoder_layers,
            mlp_layers,
            hidden_size,
            mlp_hidden_size,
            self.vocab_size,
            dropout,
            source_length,
            self.encoder_length,
            self.decoder_length,
        ).to(device)

        for i in range(iteration):
            print('Iteration {}'.format(i+1))

            train_encoder_input = train_seq_pool
            train_encoder_target = train_target_pool

            # Pre-train
            print('Pre-train EPD')
            train_controller(self.model, train_encoder_input, train_encoder_target, pretrain_epochs)
            print('Finish pre-training EPD')
            
            if self.semi:
                # Generate synthetic data
                print('Generate synthetic data for EPD')
                m = synthetic_factor * len(xtrain)
                synthetic_encoder_input, synthetic_encoder_target = self.generate_synthetic_controller_data(self.model, train_encoder_input, m,self.ss_type)
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
        jacovs = self.prepare_features(xtest, info, train=False)
        #print("test seqs:")
        for i, arch in enumerate(xtest):
            #print("jacov:{}".format(jacovs[i]))
            encoded = encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
            seq = convert_arch_to_seq(encoded['adjacency'],encoded['operations'],max_n=self.max_n)
            if self.jacov_onehot:
                jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins,one_hot=self.jacov_onehot) 
                jac_encoded = [jac +self.jacov_offset for jac in jac_encoded]
                seq.extend(jac_encoded)
            else:
                jac_encoded = discretize(jacovs[i],upper_bounds=self.jacov_bins,one_hot=self.jacov_onehot) + self.jacov_offset
                seq.append(jac_encoded)
            #print(seq)
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