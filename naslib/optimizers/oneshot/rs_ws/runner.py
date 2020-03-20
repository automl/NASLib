import inspect
import logging
import os
import pickle
import shutil
import time

from naslib.optimizers.oneshot.rs_ws.searcher import RandomNASWrapper


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)


class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung

    def to_dict(self):
        out = {'parent': self.parent, 'arch': self.arch, 'node_id':
               self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out


class RandomNAS(object):
    def __init__(self, args, search_space, save_dir):
        self.save_dir = save_dir

        self.B =  int(args.epochs * args.data_size / args.batch_size /
                      args.time_steps)
        self.model = RandomNASWrapper(args, search_space)
        self.seed = args.seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0


    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n, self.arms[n].objective_val) for n in self.arms if
                          hasattr(self.arms[n], 'objective_val')]
        objective_vals = sorted(objective_vals, key=lambda x: x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)

    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last
        # pickle save
        with open(os.path.join(self.save_dir, 'results_tmp.pkl'), 'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'),
                        os.path.join(self.save_dir, 'results.pkl'))

        self.model.save(epoch=self.model.epochs)

    def run(self):
        epochs = 0
        # self.get_eval_arch(1)
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            # If epoch has changed then evaluate the network.
            if epochs < self.model.epochs:
                epochs = self.model.epochs
                self.get_eval_arch(1)
            if self.iters % 500 == 0:
                self.save()
        self.save()

    def get_eval_arch(self, rounds=None):
        # n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1, int(self.B / 10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch()
                try:
                    ppl = self.model.evaluate(arch)
                except Exception as e:
                    ppl = 1000000
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))

            # Save sample validations
            with open(
                os.path.join(
                    self.save_dir,
                    'sample_val_architecture_epoch_{}.obj'.format(
                        self.model.epochs
                    )
                ), 'wb'
            ) as f:
                pickle.dump(sample_vals, f)

            sample_vals = sorted(sample_vals, key=lambda x: x[1])

            full_vals = []
            if 'split' in inspect.getfullargspec(self.model.evaluate).args:
                for i in range(5):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x: x[1])
                logging.info(
                    'best arch: %s, best arch valid performance: %.3f' % (
                        ' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]
                    )
                )
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])

            # Save the fully evaluated architectures
            with open(
                os.path.join(
                    self.save_dir,
                    'full_val_architecture_epoch_{}.obj'.format(
                        self.model.epochs
                    )
                ), 'wb'
            ) as f:
                pickle.dump(full_vals, f)
        return best_rounds

