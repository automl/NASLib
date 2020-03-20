import logging

import torch.nn as nn

from naslib.utils import utils
from naslib.optimizers.oneshot.base.searcher import OneShotModelWrapper


class RandomNASWrapper(OneShotModelWrapper):

    def train_batch(self, arch):
        if self.steps % len(self.train_queue) == 0:
            self.scheduler.step()
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
        lr = self.scheduler.get_lr()[0]

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        step = self.steps % len(self.train_queue)
        input, target = next(self.train_iter)

        self.model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random_ws minibatch from the search queue with replacement
        self.optimizer.zero_grad()
        logits = self.model(input, discrete=True)
        loss = self.criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        self.objs.update(loss.data.item(), n)
        self.top1.update(prec1.data.item(), n)
        self.top5.update(prec5.data.item(), n)

        if step % self.args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

        self.steps += 1
        if self.steps % len(self.train_queue) == 0:
            # Save the model weights
            self.epochs += 1
            self.train_iter = iter(self.train_queue)
            valid_err = self.evaluate(arch)
            logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1 - valid_err))
            self.save(epoch=self.epochs)

    def evaluate(self, arch, split=None):
        # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        self.model.eval()

        if split is None:
            n_batches = 10
        else:
            n_batches = len(self.valid_queue)

        for step in range(n_batches):
            try:
                input, target = next(self.valid_iter)
            except Exception as e:
                logging.info('looping back over valid set')
                self.valid_iter = iter(self.valid_queue)
                input, target = next(self.valid_iter)
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=True)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1 - 0.01 * top1.avg

    def evaluate_test(self, arch, split=None, discrete=False, normalize=True):
        # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        self.model.eval()

        if split is None:
            n_batches = 10
        else:
            n_batches = len(self.test_queue)

        for step in range(n_batches):
            try:
                input, target = next(self.test_iter)
            except Exception as e:
                logging.info('looping back over valid set')
                self.test_iter = iter(self.test_queue)
                input, target = next(self.test_iter)
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=discrete, normalize=normalize)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1 - 0.01 * top1.avg

