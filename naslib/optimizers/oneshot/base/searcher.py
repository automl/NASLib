import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable

from naslib.utils import utils
from naslib.optimizers.oneshot.base.model_search import Network


class OneShotModelWrapper(object):
    def __init__(self, args, search_space, resume_epoch=None):
        """Wrapper class for the one-shot model instantiation, training and evaluation.
        :args: arguments
        :search_space: SearchSpace object
        """

        self.args = args
        self.search_space = search_space

        # Dump the config of the run, but if only if it doesn't yet exist
        config_path = os.path.join(args.save, 'config.json')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as fp:
                json.dump(args.__dict__, fp)
        self.seed = args.seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled = True
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        model = Network(args.init_channels, 10, args.layers, self.criterion, output_weights=args.output_weights,
                        search_space=search_space, steps=search_space.num_intermediate_nodes)

        model = model.cuda()
        self.model = model

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        #TODO: add option to load checkpoint
        if resume_epoch is not None:
            self.epoch = int(resume_epoch)
            logging.info("Resuming from epoch %d" % self.epoch)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epoch):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train(self, epoch, lr, architect=None):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(self.train_queue):
            self.model.train()
            n = input.size(0)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # get a minibatch from the search queue with replacement
            try:
                input_search, target_search = next(self.valid_iter)
            except:
                self.valid_iter = iter(self.valid_queue)
                input_search, target_search = next(self.valid_iter)

            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            # Allow for warm starting of the one-shot model for more reliable architecture updates.
            if architect is not None:
                if epoch >= self.args.warm_start_epochs:
                    architect.step(input_train=input,
                                   target_train=target,
                                   input_valid=input_search,
                                   target_valid=target_search,
                                   eta=lr,
                                   network_optimizer=self.optimizer,
                                   unrolled=self.args.unrolled)

            self.optimizer.zero_grad()
            logits = self.model(input)
            loss = self.criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def infer(self, epoch, discrete=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        self.model.eval()

        for step, (input, target) in enumerate(self.valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=discrete)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def save(self, epoch):
        utils.save(self.model, os.path.join(self.args.save, 'one_shot_model_{}.pt'.format(epoch)))

    def load(self, epoch=None):
        if epoch is not None:
            model_obj_path = os.path.join(self.args.save, 'one_shot_model_{}.obj'.format(epoch))
            if os.path.exists(model_obj_path):
                utils.load(self.model, model_obj_path)
            else:
                model_pt_path = os.path.join(self.args.save, 'one_shot_model_{}.pt'.format(epoch))
                utils.load(self.model, model_pt_path)
        else:
            utils.load(self.model, os.path.join(self.args.save, 'weights.obj'))

    def get_weights_from_arch(self, arch):
        adjacency_matrix, node_list = arch
        num_ops = len(self.search_space._PRIMITIVES)

        # Assign the sampled ops to the mixed op weights.
        # These are not optimized
        alphas_mixed_op = Variable(torch.zeros(self.model._steps, num_ops).cuda(), requires_grad=False)
        for idx, op in enumerate(node_list):
            alphas_mixed_op[idx][self.search_space._PRIMITIVES.index(op)] = 1

        # Set the output weights
        alphas_output = Variable(torch.zeros(1, self.model._steps + 1).cuda(), requires_grad=False)
        for idx, label in enumerate(list(adjacency_matrix[:, -1][:-1])):
            alphas_output[0][idx] = label

        # Initialize the weights for the inputs to each choice block.
        if str(self.model.search_space) == 'SearchSpace1':
            begin = 3
        else:
            begin = 2
        alphas_inputs = [Variable(torch.zeros(1, n_inputs).cuda(), requires_grad=False) for n_inputs in
                         range(begin, self.model._steps + 1)]
        for alpha_input in alphas_inputs:
            connectivity_pattern = list(adjacency_matrix[:alpha_input.shape[1], alpha_input.shape[1]])
            for idx, label in enumerate(connectivity_pattern):
                alpha_input[0][idx] = label

        # Total architecture parameters
        arch_parameters = [
            alphas_mixed_op,
            alphas_output,
            *alphas_inputs
        ]
        return arch_parameters

    def set_arch_model_weights(self, weights):
        self.model._arch_parameters = weights

    def sample_arch(self):
        adjacency_matrix, op_list = self.search_space.sample(with_loose_ends=True, upscale=False)
        return adjacency_matrix, op_list

