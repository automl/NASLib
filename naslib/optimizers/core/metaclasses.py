from abc import ABCMeta
from abc import abstractmethod


class MetaOptimizer(metaclass=ABCMeta):
    """
    Abstract class for all NAS optimizers.
    """

    using_step_function = True

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
        if self.using_step_function:
            raise NotImplementedError()

    def train_statistics(self):
        """
        If the step function is not used we need the statistics from
        the optimizer
        """
        if not self.using_step_function:
            raise NotImplementedError()

    def test_statistics(self):
        """
        Return anytime test statistics if provided by the optimizer
        """
        pass

    @abstractmethod
    def adapt_search_space(self, search_space, dataset, scope=None):
        """
        Modify the search space to fit the optimizer's needs,
        e.g. discretize, add architectural parameters, ...

        To modify the search space use `search_space.update(...)`

        Good practice is to deepcopy the search space, store
        the modified version and leave the original search space
        untouched in case it is beeing used somewhere else.

        Args:
            search_space (Graph): The search space we are doing NAS in.
            dataset (str): String representation of the used dataset
            scope (str or list(str)): The scope of the search space which
                should be optimized by the optimizer.
        """
        raise NotImplementedError()

    def new_epoch(self, epoch):
        """
        Function called at the beginning of each new search epoch. To be
        used as hook for the optimizer.

        Args:
            epoch (int): Number of the epoch to start.
        """
        pass

    def before_training(self):
        """
        Function called right before training starts. To be used as hook
        for the optimizer.
        """
        pass

    def after_training(self):
        """
        Function called right after training finished. To be used as hook
        for the optimizer.
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

    def get_model_size(self):
        """
        Returns the size of the model parameters in mb, e.g. by using
        `utils.count_parameters_in_MB()`.

        This is only used for logging purposes.
        """
        return 0

    def get_checkpointables(self):
        """
        Return all objects that should be saved in a checkpoint during training.

        Will be called after `before_training` and must include key "model".

        Returns:
            (dict): with name as key and object as value. e.g. graph, arch weights, optimizers, ...
        """
        pass
