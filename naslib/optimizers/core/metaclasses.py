from abc import ABCMeta, abstractmethod

import six
import torch.nn as nn


@six.add_metaclass(ABCMeta)
class MetaOp(nn.Module):
    def __init__(self, primitives, *args, **kwargs):
        super(MetaOp, self).__init__()
        self.primitives = primitives
        self._ops = nn.ModuleList()

    def __len__(self):
        return len(self.primitives)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build(self, C, stride, out_node_op, ops_dict):
        pass


class MetaOptimizer(object, metaclass=ABCMeta):
    """
    Abstract class for all NAS optimizers.
    """

    
    @abstractmethod
    def step(self, data_train, data_val):
        """
        Run one optimizer step with the batch of training and test data.

        Args:
            data_train (tuple(Tensor, Tensor)): A tuple of input and target
                tensors from the training split
            data_val (tuple(Tensor, Tensor)): A tuple of input and target
                tensors from the validation split
            error_dict

        Returns:
            dict: A dict containing training statistics (TODO)
        """
        raise NotImplementedError()


    @abstractmethod
    def adapt_search_space(self, search_space, scope=None):
        """
        Modify the search space to fit the optimizer's needs,
        e.g. discretize, add architectural parameters, ...

        To modify the search space use `search_space.update(...)`

        Good practice is to deepcopy the search space, store
        the modified version and leave the original search space
        untouched in case it is beeing used somewhere else.

        Args:
            search_space (Graph): The search space we are doing NAS in.
            scope (str or list(str)): The scope of the search space which
                should be optimized by the optimizer.
        """
        raise NotImplementedError()


    def new_epoch(self, epoch):
        """
        Function called at the beginning of each new search epoch

        Args:
            epoch (int): Number of the epoch to start.
        """
        pass


    @abstractmethod
    def get_final_architecture(self):
        """
        Returns the final discretized architecture.

        Returns:
            Graph: The final architecture.
        """
        raise NotImplementedError()


    @abstractmethod
    def get_op_optimizer(self):
        """
        This is required for the final validation when
        training from scratch.

        Returns:
            (torch.optim.Optimizer): The optimizer used for the op weights update.
        """

    #@abstractmethod
    #def replace_function(self, edge, graph):
    #    raise NotImplementedError


    #@abstractmethod
    @classmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError

    #@abstractmethod
    def forward_pass_adjustment(self, *args, **kwargs):
        """
        Function evaluated prior to every forward pass
        """
        raise NotImplementedError

