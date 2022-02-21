from distutils.command.config import config
import logging

import torch
from collections.abc import Iterable

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops
from naslib.optimizers.oneshot.gsparsity.ProxSGD_for_groups import ProxSGD

import numpy as np

logger = logging.getLogger(__name__)


class GSparseOptimizer(MetaOptimizer):
    """
    Implements Group Sparsity as defined in
        Chatzimichailidis et. al. : 
        GSparsity: Unifying Network Pruning and 
        Neural Architecture Search by Group Sparsity
    """
    mu=0
    def __init__(
        self,
        config,
        op_optimizer: torch.optim.Optimizer = ProxSGD,    
        op_optimizer_evaluate: torch.optim.Optimizer = torch.optim.SGD,     
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer

        Group sparsity paper uses ProxSGD for optimizing operation weights
            during search phase.
        And SGD for optimizing weights during evaluation phase.

        Args:
            epochs (int): Number of epochs. Required for tau
            mu (float): corresponds to the Weight decay
            threshold (float): threshold of pruning
            op_optimizer (torch.optim.Optimizer: ProxSGD): optimizer for the op weights 
            op_optmizer_evaluate: (torch.optim.Optimizer): optimizer for the op weights            
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super(GSparseOptimizer, self).__init__()

        self.config = config
        self.op_optimizer = op_optimizer
        self.op_optimizer_evaluate = op_optimizer_evaluate
        self.loss = loss_criteria
        self.dataset = config.dataset
        self.grad_clip = config.search.grad_clip
        self.mu = config.search.weight_decay
        mu = self.mu
        self.threshold = config.search.threshold
        self.normalization_exponent = config.search.normalization_exponent
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
           torch.ones(size=[len_primitives], requires_grad=False), requires_grad=False
        )
        weights = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[0.0]), requires_grad=False
        )
        weight_decay = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[GSparseOptimizer.mu])#, requires_grad=False)
        )
        edge.data.set("alpha", alpha, shared=True)
        edge.data.set("weights", weights, shared=True)
        
        for i in range(len(edge.data.op)):
            try:
                len(edge.data.op[i].op)
            except Exception:
                weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
                edge.data.op[i].register_parameter('weight', weight)

        #import ipdb;ipdb.set_trace()
        #edge.data.set("weight_decay", weight_decay, shared=False)

    def adapt_search_space(self, search_space, scope=None, **kwargs):
        """
        Modify the search space to fit the optimizer's needs,
        e.g. discretize, add alpha flag and group name, ...
        Args:
            search_space (Graph): The search space we are doing NAS in.
            scope (str or list(str)): The scope of the search space which
                should be optimized by the optimizer.
        """
        self.search_space = search_space
        graph = search_space.clone()

        # If there is no scope defined, let's use the search space default one
        if not scope:
            scope = graph.OPTIMIZER_SCOPE

        # 1. add alpha flags for pruning
        graph.update_edges(
            self.__class__.add_alphas, scope=scope, private_edge_data=False
        )

        # 2. replace primitives with mixed_op
        graph.update_edges(
            self.__class__.update_ops, scope=scope, private_edge_data=True
        )

        graph.parse()

        # initializing the ProxSGD optmizer for the operation weights
        #import ipdb;ipdb.set_trace()
        self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=self.mu,
            clip_bounds=(0,1),
            normalization=self.config.search.normalization,
            normalization_exponent=self.normalization_exponent
        )

        graph.train()

        self.graph = graph
        self.scope = scope
    
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
        input_train, target_train = data_train
        input_val, target_val = data_val

        self.graph.train()
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        #import ipdb;ipdb.set_trace()
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        with torch.no_grad():
            self.graph.eval()
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)
        
        return logits_train, logits_val, train_loss, val_loss
    
    def get_final_architecture(self):
        """
        Returns the final discretized architecture.

        Returns:
            Graph: The final architecture.
        """
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()

        def update_l2_weights(edge):
            if edge.data.has("alpha"):
                #primitives = edge.data.op.get_embedded_ops()
                weight=0.0
                for i in range(len(edge.data.op.primitives)):
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:
                                weight+= (torch.norm(edge.data.op.primitives[i].op[j].weight.cpu()))**2
                            except Exception:
                                continue
                        edge.data.weights[i]+=weight                            
                        weight=0.0
                    except Exception:                        
                        #import ipdb;ipdb.set_trace()
                        try:
                            edge.data.weights[i]+=(edge.data.op.primitives[i].weight.cpu())**2
                        except Exception:
                            continue
                    
        def prune_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.weights)):
                    if torch.sqrt(edge.data.weights[i]) < self.threshold:
                        edge.data.alpha[i]=0

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                #print(alphas)
                #import ipdb;ipdb.set_trace()
                #edge.head
                #edge.tail
                #graph.add_node(new_node#>1)
                #graph.add_edges(edge.head, new_node#)
                #graph.add_edges(new_node#, edge.tail)
                #graph.edges(edge.head, new_node#).set("op", the operation)
                #graph.edges(new_node#, edge.tail).set("op", Identity())
                edge.data.set("op", primitives[np.argmax(alphas)])

        graph.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)
        graph.update_edges(prune_weights, scope=self.scope, private_edge_data=True)
        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        graph = graph.to(self.device)
        return graph

    def test_statistics(self):
        """
        Return anytime test statistics if provided by the optimizer
        """
        # nb301 is not there but we use it anyways to generate the arch strings.
        # if self.graph.QUERYABLE:
        try:
            # record anytime performance
            best_arch = self.get_final_architecture()
            return best_arch.query(Metric.TEST_ACCURACY, self.dataset)
        except:
            return None

    def before_training(self):
        """
        Function called right before training starts. To be used as hook
        for the optimizer.
        """
        """
        Move the graph into cuda memory if available.
        """
        self.graph = self.graph.to(self.device)


    def get_op_optimizer(self):
        """
        This is required for the final validation when
        training from scratch.

        Returns:
            (torch.optim.Optimizer): The optimizer used for the op weights update.
        """
        return self.op_optimizer_evaluate.__class__

    def get_model_size(self):
        return count_parameters_in_MB(self.graph)

    def get_checkpointables(self):
        """
        Return all objects that should be saved in a checkpoint during training.

        Will be called after `before_training` and must include key "model".

        Returns:
            (dict): with name as key and object as value. e.g. graph, arch weights, optimizers, ...
        """
        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "op_optimizer_evaluate": self.op_optimizer_evaluate,            
        }

class GSparseMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for Group Sparsity.

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self._ops = torch.nn.ModuleList()
        
        #for primitive in primitives:
            #op = 
        self.min_cuda_memory = min_cuda_memory

    def forward(self, x, edge_data):
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """
        # sampled_arch_weight = edge_data.sampled_arch_weight

        summed = sum(op(x, None) for op in self.primitives)
        #print(summed)
        
        summed = torch.nn.functional.normalize(summed)
        #print(summed)
        #import ipdb;ipdb.set_trace()
        #import ipdb;ipdb.set_trace()

        return summed