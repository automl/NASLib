from dataclasses import replace
from distutils.command.config import config
from locale import normalize
import logging
from turtle import pos, position
from matplotlib.colors import NoNorm
import torch.nn.utils.parametrize as P
import torch
from collections.abc import Iterable

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

from naslib.optimizers.oneshot.gsparsity.ProxSGD_for_groups import ProxSGD
import naslib.search_spaces.core.primitives as primitives

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
        self.threshold = config.search.threshold
        self.normalization = config.search.normalization
        self.normalization_exponent = config.search.normalization_exponent
        self.operation_weights = torch.nn.ParameterList()
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
        And add a parameter 'weights' that will be used for storing the l2 norm
        of the weights of the operations which later is used for pruning.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
           torch.zeros(size=[len_primitives], requires_grad=False), requires_grad=False
        )
        weights = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[0.0]), requires_grad=False
        )
        dimension = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[0.0]), requires_grad=False
        )
        edge.data.set("alpha", alpha, shared=True)
        edge.data.set("weights", weights, shared=True)
        edge.data.set("dimension", weights, shared=True)

    @staticmethod
    def add_weights(edge):
        """
        Operations like Identity(), Zero(stride=1) etc do not have weights of their own, 
        neither contained suboperations that have weights, just to optimize over such operations
        we attach a 'weight' parameter to them, which is used in the forward() of the MixedOp
        thus updating them and optimizing over them.
        IMPORTANT: In GroupSparsity, suboperation that do not have weights are ignored from 
        optimization point of view, i.e. they are not given weights explicitely to be used while
        calculating the weight of the operation containing them.
        """
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(len(edge.data.op)):
            try:                
                len(edge.data.op[i].op)                
            except AttributeError:
                weight = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
                edge.data.op[i].register_parameter("weight", weight)
    

    def adapt_search_space(self, search_space, scope=None, **kwargs):
        """
        Modify the search space to fit the optimizer's needs,
        e.g. discretize, add alpha flag and shared weight parameter, ...
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

        # 2. add weight parameter to operations without weight
        graph.update_edges(
            self.__class__.add_weights, scope=scope, private_edge_data=True
        )

        # 3. replace primitives with mixed_op
        graph.update_edges(
            self.__class__.update_ops, scope=scope, private_edge_data=True
        )

        for alpha in graph.get_all_edge_data("weights"):
            self.operation_weights.append(alpha)    

        graph.parse()
        print(graph)

        # initializing the ProxSGD optmizer for the operation weights        
        self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=self.mu,
            clip_bounds=(0,1),
            normalization=self.normalization,
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
            dict: A dict containing training statistics
        """
        input_train, target_train = data_train
        input_val, target_val = data_val

        self.graph.train()
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()#retain_graph=True)
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
        normalization_exponent=self.normalization_exponent
        def update_l2_weights(edge):
            """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of 
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.
            """            
            if edge.data.has("alpha"):
                weight=0.0
                group_dim=torch.zeros(1)
                for i in range(len(edge.data.op.primitives)):
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:                                
                                group_dim += torch.numel(edge.data.op.primitives[i].op[j].weight)
                                weight+= (torch.norm(edge.data.op.primitives[i].op[j].weight,2)**2).item()
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(len(edge.data.op.primitives[i].op[j].op)):                                
                                        group_dim += torch.numel(edge.data.op.primitives[i].op[j].op[k].weight)
                                        weight+= (torch.norm(edge.data.op.primitives[i].op[j].op[k].weight,2)**2).item()
                                except AttributeError:
                                    continue                         
                        edge.data.weights[i]+=weight
                        edge.data.dimension[i]+=group_dim.item()                          
                        weight=0.0
                        group_dim=torch.zeros(1)
                    except AttributeError:   
                        size=torch.tensor(torch.numel(edge.data.op.primitives[i].weight))                        
                        edge.data.weights[i]+=(edge.data.op.primitives[i].weight.item())**2
                        edge.data.dimension[i]+=size
        
        def normalize_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.op.primitives)):
                    edge.data.weights[i]=torch.sqrt(edge.data.weights[i])/torch.pow(edge.data.dimension[i], normalization_exponent).item()

                    
        def prune_weights(edge):
            """
            Operations whose l2 norm of the weights across all cells of the same
            type (normal or reduced) is less than the threshold are pruned away.
            To achieve this, the alpha flag for the corresponding operation is
            turned off (replaced with zero).
            """
            if edge.data.has("alpha"):                
                for i in range(len(edge.data.weights)):
                    if torch.sqrt(edge.data.weights[i]) < self.threshold:
                        edge.data.alpha[i]=0 
        
        def reinitialize_l2_weights(edge):
            if edge.data.has("alpha"):                
                for i in range(len(edge.data.weights)):
                    edge.data.weights[i]=0
                    edge.data.dimension[i]=0

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                """
                The next 2 lines of code is just to make sure only 1 operation is chosen per edge
                so that the resulting architecture is comparable to other optimizers and 
                queryable from the benchmark.
                """
                weights= edge.data.weights.detach().cpu()
                alphas = torch.nn.Parameter(torch.zeros(size=[len(alphas)], requires_grad=False), requires_grad=False)
                alphas[torch.argmax(weights)]=1
                """
                Only the operations whose alpha are non-zero are retained,
                others are pruned away. If on an edge, more than 1 operations 
                are to be retained, then the operation of the edge is set to a MixedOp
                of these operations.
                """
                positions = alphas.nonzero()                
                if len(positions)>1:
                    operations=[]
                    for pos in positions:
                        operations.append(primitives[pos])
                    edge.data.set("op", GSparseMixedOp(operations))
                else:                    
                    edge.data.set("op", primitives[positions.item()])

        # Detailed description of the operations are provided in the functions.
        graph.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)        
        graph.update_edges(normalize_weights, scope=self.scope, private_edge_data=True)
        #graph.update_edges(prune_weights, scope=self.scope, private_edge_data=True)
        
        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.update_edges(reinitialize_l2_weights, scope=self.scope, private_edge_data=False)
        graph.prepare_evaluation()
        graph.parse()
        #graph.QUERYABLE=False
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
        self.operation_weights = self.operation_weights.to(self.device)

    def new_epoch(self, epoch):
        """
        Just log the l2 norms of operation weights.
        """
        normalization_exponent=self.normalization_exponent
        def update_l2_weights(edge):
            """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of 
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.
            """            
            if edge.data.has("alpha"):
                weight=0.0
                group_dim=torch.zeros(1)
                for i in range(len(edge.data.op.primitives)):
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:
                                group_dim += torch.numel(edge.data.op.primitives[i].op[j].weight)
                                weight+= (torch.norm(edge.data.op.primitives[i].op[j].weight,2)**2).item()
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(len(edge.data.op.primitives[i].op[j].op)):
                                        group_dim += torch.numel(edge.data.op.primitives[i].op[j].op[k].weight)
                                        weight+= (torch.norm(edge.data.op.primitives[i].op[j].op[k].weight,2)**2).item()
                                except AttributeError:
                                    continue                         
                        edge.data.weights[i]+=weight
                        edge.data.dimension[i]+=group_dim.item()                          
                        weight=0.0
                        group_dim=torch.zeros(1)
                    except AttributeError:   
                        size=torch.tensor(torch.numel(edge.data.op.primitives[i].weight))                        
                        edge.data.weights[i]+=(edge.data.op.primitives[i].weight.item())**2
                        edge.data.dimension[i]+=size
        
        def normalize_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.op.primitives)):
                    edge.data.weights[i]=edge.data.weights[i]/torch.pow(edge.data.dimension[i], normalization_exponent).item()
        
        def reinitialize_l2_weights(edge):
            if edge.data.has("alpha"):                
                for i in range(len(edge.data.weights)):                    
                    edge.data.weights[i]=0
                    edge.data.dimension[i]=0

        self.graph.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)
        self.graph.update_edges(normalize_weights, scope=self.scope, private_edge_data=True)  

        for alpha in self.graph.get_all_edge_data("weights"):            
            self.operation_weights.append(alpha)        
        weights_str = [
            ", ".join(["{:+.06f}".format(torch.sqrt(x)) for x in a])
            + ", {}".format(np.max(torch.sqrt(a).detach().cpu().numpy()))
            for a in self.operation_weights
        ]
        logger.info(
            "Arch weights (normalized weights, last column max): \n{}".format(
                "\n".join(weights_str)
            )
        )
        self.graph.update_edges(reinitialize_l2_weights, scope=self.scope, private_edge_data=False)
        self.operation_weights = torch.nn.ParameterList()
        super().new_epoch(epoch)

    def after_training(self):
        print("save path: ", self.config.save)
        best_arch = self.get_final_architecture()
        logger.info("Final architecture after search:\n" + best_arch.modules_str())


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
        self.min_cuda_memory = min_cuda_memory

    def forward(self, x, edge_data):
        """
        Output of operations like Identity(), that do not have weighted suboperations
        like Conv2d(), are multipled with the weight parameter attached to them, so 
        that these weights are optimized as well, during the training phase.
        """        
        summed=0
        for op in self.primitives:
            try:                
                len(op.op)  
                summed+=op(x, None)
            except AttributeError:                
                if op.training and edge_data.has("alpha"):
                    summed+=op.weight*op(x,None)
                else:
                    summed+=op(x, None)
        
        summed = torch.nn.functional.normalize(summed)
        return summed
