import logging
import torch
import torch.nn as nn

from naslib.optimizers.core import Evaluator


class Searcher(Evaluator):
    def __init__(self, graph, parser, arch_optimizer, *args, **kwargs):
        super(Searcher, self).__init__(graph, parser, *args, **kwargs)
        self.arch_optimizer = arch_optimizer
        self.arch_optimizer.architectural_weights.to(self.device)
        self.run_kwargs['arch_optimizer'] = self.arch_optimizer


    def run(self, n_evaluations, *args, **kwargs):
        for n in range(n_evaluations):
            logging.info('Iteration: %d'%n)
            self.arch_optimizer.step()

            if hasattr(self.graph, 'query_architecture'):
                # Record anytime performance
                arch_info = self.graph.query_architecture(self.arch_optimizer.architectural_weights)
                logging.info('arch {}'.format(arch_info))
                if 'arch_eval' not in self.errors_dict:
                    self.errors_dict['arch_eval'] = []
                self.errors_dict['arch_eval'].append(arch_info)
                self.log_to_json(self.parser.config.save)
            else:
                train_acc, train_obj, runtime = self.train(
                    self.parser.config.epochs,
                    self.model,
                    self.optimizer,
                    self.criterion,
                    self.train_queue,
                    self.valid_queue,
                    device=self.device,
                    **self.run_kwargs
                )
                logging.info('train_acc %f', train_acc)

                if len(self.valid_queue) != 0:
                    valid_acc, valid_obj = self.infer(
                        self.model,
                        self.criterion,
                        self.valid_queue,
                        device=self.device
                    )
                    logging.info('valid_acc %f', valid_acc)
                    self.errors_dict.valid_acc.append(valid_acc)
                    self.errors_dict.valid_loss.append(valid_obj)

                test_acc, test_obj = self.infer(self.model, self.criterion,
                                                self.test_queue, device=self.device)
                logging.info('test_acc %f', test_acc)

                self.errors_dict.train_acc.append(train_acc)
                self.errors_dict.train_loss.append(train_obj)
                self.errors_dict.test_acc.append(test_acc)
                self.errors_dict.test_loss.append(test_obj)
                self.errors_dict.runtime.append(runtime)
                self.log_to_json(self.parser.config.save)

