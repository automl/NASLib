import logging

import torch

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops
import ProxSGD_for_groups as ProxSGD
import utils_sparsenas

logger = logging.getLogger(__name__)


class GSparseOptimizer(MetaOptimizer):
    """
    Implements Group Sparsity as defined in
        # Add name of authors
        GSparsity: Unifying Network Pruning and 
        Neural Architecture Search by Group Sparsity
    """
    def __init__(
        self,
        config,
        op_optimizer: torch.optim.Optimizer = torch.optim.SGD,        
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer

        Args:
            epochs (int): Number of epochs. Required for tau
            mu (float): corresponds to the Weight decay
            threshold (float): threshold of pruning
            op_optimizer (torch.optim.Optimizer): optimizer for the op weights            
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super(GSparseOptimizer, self).__init__()

        self.config = config
        self.op_optimizer = op_optimizer
        self.loss = loss_criteria
        self.dataset = config.dataset
        self.grad_clip = config.search.grad_clip
        self.threshold = config.search.threshold
        self.mu = config.search.mu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GSparse specific GSparseMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GSparseMixedOp(primitives))
    
    @staticmethod
    def add_alphas(edge):
        """
        Function to add the pruning flag 'alpha' to the edges.
        And initialize a group name for all primitives.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
           torch.ones(size=[len_primitives], requires_grad=False)
        )
        group = torch.nn.Parameter(
           torch.zeros(size=[len_primitives], requires_grad=False)
        )
        edge.data.set("alpha", alpha, shared=True)
        edge.data.set("group", group, shared=True)
    
    def group_primitives(graph):
        """
        Function to group similar operations together
        A group could be the same operation in all cells of the same type (normal or reduce). 
        For example, one group is op 3 of edge 7 in Cells 0-1/3-4/5-7 (normal cells), 
        another group is op 3 of edge 7 in Cell 2/5 (reduce cells).
    
        A group could also be the same operation in all cells in a stage (there may be multiple stages). 
        For example, one group is op 3 of edge 7 in all cells of stage_normal_1 (Cells 0-1), 
        another group is op 3 of edge 7 in all cells of stage_reduce_1 (Cell 2), 
        another group is op 3 of edge 7 in all cells of stage_normal_2 (Cells 3-4), 
        another group is op 3 of edge 7 in all cells of stage_reduce_2 (Cell 5), 
        one group is op 3 of edge 7 in all cells of stage_normal_3 (Cells 6-7).
    
        """
        #makrograph-subgraph_at(4).normal_cell-edge(1,6).primitive-6.op.1.weight'
        #graph.nodes[5]['subgraph'].edges[1, 6]['op'].primitives[6].op
        #graph.nodes[5]['subgraph'].edges[1, 6]
        #graph.nodes[5]['subgraph'].scope

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




class GSparseMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for Group Sparsity.

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def forward(self, x, edge_data):
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """
        # sampled_arch_weight = edge_data.sampled_arch_weight

        summed = sum(op(x, None) for op in self.primitives)

        return summed