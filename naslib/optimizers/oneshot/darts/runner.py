import os
import json
import logging
import pickle
import torch

from naslib.optimizers.oneshot.darts.searcher import DARTSWrapper
from naslib.optimizers.oneshot.darts.architect import Architect
from naslib.utils import utils


class DARTS(object):
    def __init__(self, args, search_space):
        self.args = args
        self.model = DARTSWrapper(args, search_space)
        self.architect = Architect(self.model.model,
                                   args.momentum,
                                   args.weight_decay,
                                   args.arch_learning_rate,
                                   args.arch_weight_decay)

    def run(self):

        for epoch in range(self.args.epochs):
            self.model.scheduler.step()
            lr = self.model.scheduler.get_lr()[0]
            # increase the cutout probability linearly throughout search
            self.model.train_transform.transforms[-1].cutout_prob = self.args.cutout_prob * epoch / (self.args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         self.model.train_transform.transforms[-1].cutout_prob)

            # Save the one shot model architecture weights for later analysis
            arch_filename = os.path.join(self.args.save,
                                         'one_shot_architecture_{}.obj'.format(epoch))
            with open(arch_filename, 'wb') as filehandler:
                numpy_tensor_list = []
                for tensor in self.model.model.arch_parameters():
                    numpy_tensor_list.append(tensor.detach().cpu().numpy())
                pickle.dump(numpy_tensor_list, filehandler)

            # Save the entire one-shot-model
            filepath = os.path.join(self.args.save,
                                    'one_shot_model_{}.obj'.format(epoch))
            torch.save(self.model.model.state_dict(), filepath)

            #logging.info('architecture', str(numpy_tensor_list))

            # training
            train_acc, train_obj = self.model.train(epoch, lr, architect=self.architect)
            logging.info('train_acc %f', train_acc)

            # validation
            valid_acc, valid_obj = self.model.infer(epoch)
            logging.info('valid_acc %f', valid_acc)

            utils.save(self.model.model, os.path.join(self.args.save, 'weights.pt'))

