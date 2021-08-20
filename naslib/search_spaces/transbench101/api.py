import os
import json
import copy
import torch
import random


class TransNASBenchAPI(object):
    """
    This is the class for accessing raw information stored in the .pth file.
    """

    def __init__(self, database_path, verbose=False):
        self.database = torch.load(database_path)
        self.verbose = verbose

        self.metrics_dict = self.database['metrics_dict']  # {task : metrics_list}
        self.info_names = self.database['info_names']  # ['inference_time', 'encoder_params', 'encoder_FLOPs' ... ]
        self.task_list = self.database['task_list']  # [7 tasks]

        self.search_spaces = list(self.database['data'].keys())
        self.all_arch_dict = {k: list(self.database['data'][k].keys()) for k in self.search_spaces}
        self.arch2space = self._gen_arch2space(self.all_arch_dict)

        self.data = self._gen_all_data(self.database, self.arch2space)

    def __getitem__(self, index):
        arch = self.index2arch(index)
        return copy.deepcopy(self.get_arch_result(arch))

    def __len__(self):
        return len(self.arch2space.keys())

    def __repr__(self):
        return ('{name}({total} architectures/{task} tasks)'.format(name=self.__class__.__name__,
                                                                    total=len(self), task=len(self.task_list)))

    def index2arch(self, index):
        return list(self.arch2space.keys())[index]

    def arch2index(self, arch):
        return list(self.arch2space.keys()).index(arch)

    def get_arch_result(self, arch):
        return self.data[self.arch2space[arch]][arch]

    def get_total_epochs(self, arch, task):
        arch_result = self.get_arch_result(arch)
        return arch_result[task]['total_epochs']

    def get_model_info(self, arch, task, info):
        assert info in self.info_names, f"info {info} is not available! Must in {self.info_names}!"
        arch_result = self.get_arch_result(arch)
        return arch_result[task]['model_info'][info]

    def get_single_metric(self, arch, task, metric, mode="best", xseed=None):
        """
        get single metric value
        Args:
            arch: architecture string
            task: a single task in tasks specified in self.task_list
            metric: the metric name for querying
            mode: ['final', 'best', 'list'] or epoch_number
            xseed: [None] or seed number

        Returns:
            metric value or values according to mode of querying
        """
#         print('metric ----->', metric)
#         print('metrics_dict ----->', self.metrics_dict[task])
        assert metric in self.metrics_dict[task], \
            f"metric {metric} is not available for task {task}! Must in {self.metrics_dict[task]}!"
        arch_result = self.get_arch_result(arch)
        metric_list = arch_result[task]['metrics'][metric]

        if isinstance(mode, str):
            if mode == 'final':
                return metric_list[-1]
            elif mode == 'best':
                return max(metric_list)
            elif mode == 'list':
                return metric_list
            else:
                raise ValueError(f"get_metric() str mode can only be ['final', 'best', 'list']")
        elif isinstance(mode, int):
            assert mode < len(
                metric_list), f"get_metric() int mode must < total epoch {len(metric_list)} for task {task}!"
            return metric_list[mode]
        else:
            raise ValueError(f"get_metric() mode must be 'final', 'best', 'list' or epoch_number")

    def get_epoch_status(self, arch, task, epoch, xseed=None):
        assert isinstance(epoch, int), f"arg epoch {epoch} must be int"
        arch_result = self.get_arch_result(arch)
        epoch_upper = arch_result[task]['total_epochs']
        assert epoch < epoch_upper, f"arg epoch {epoch} must < {epoch_upper} on task {task}"

        exp_dict = arch_result[task]['metrics']

        ep_status = {'epoch': epoch if epoch >= 0 else self.get_total_epochs(arch, task) + epoch}
        ep_status = {**ep_status, **{k: exp_dict[k][epoch] for k in self.metrics_dict[task]}}
        return ep_status

    def get_best_epoch_status(self, arch, task, metric):
        """
        get the best epoch status with respect to a certain metric (equiv. to early stopping at best validation metric)
        Args:
            arch: architecture string
            task: task name
            metric: metric name specified in the metrics_dict

        Returns: a status dict of the best epoch
        """
        assert metric in self.metrics_dict[task], \
            f"metric {metric} is not available for task {task}! Must in {self.metrics_dict[task]}!"
        metric_list = self.get_single_metric(arch, task, metric, mode="list")
        best_epoch = max(range(len(metric_list)), key=lambda i: metric_list[i])  # return argmax
        best_ep_status = self.get_epoch_status(arch, task, epoch=best_epoch)
        return best_ep_status

    def get_arch_list(self, search_space):
        assert search_space in self.search_spaces, f'search_space must in {self.search_spaces}'
        return self.all_arch_dict[search_space]

    def get_best_archs(self, task, metric, search_space, topk=1):
        arch_list = self.get_arch_list(search_space=search_space)
        tuple_list = list(map(lambda arch: (self.get_single_metric(arch, task, metric, mode="best"), arch), arch_list))
        return sorted(tuple_list, reverse=True)[:topk]

    def _gen_arch2space(self, all_arch_dict):
        result = {}
        for ss, ls in all_arch_dict.items():
            tmp = dict(zip(ls, [ss] * len(ls)))
            result = {**result, **tmp}
        return result

    def _gen_all_data(self, database, arch2space):
        data = {}
        for ss in self.search_spaces:
            data[ss] = {}
        for idx, (arch, space) in enumerate(arch2space.items()):
            data[space][arch] = ArchResult(idx, arch, database['data'][space][arch])
        return data


class ArchResult(object):

    def __init__(self, arch_index, arch_str, all_results):
        self.arch_index = int(arch_index)
        self.arch_str = copy.deepcopy(arch_str)

        assert isinstance(all_results, dict)
        self.all_results = all_results

    def __repr__(self):
        return (
            '{name}(arch-index={index}, arch={arch}, {num} tasks)'.format(name=self.__class__.__name__,
                                                                          index=self.arch_index,
                                                                          arch=self.arch_str,
                                                                          num=len(self.all_results)))

    def __getitem__(self, item):
        return self.all_results[item]

    def query_all_results(self):
        return self.all_results
