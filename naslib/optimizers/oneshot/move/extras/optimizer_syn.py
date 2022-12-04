from dataclasses import replace
from distutils.command.config import config
from importlib.resources import path
from locale import normalize
import logging
import math
from tokenize import group
from turtle import pos, position
from attr import has
from matplotlib.colors import NoNorm
import torch.nn.utils.parametrize as P
import torch
import torch.nn.functional as F
from collections.abc import Iterable

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

from naslib.optimizers.oneshot.gsparsity.ProxSGD_for_groups import ProxSGD
import naslib.search_spaces.core.primitives as primitives

import copy
import numpy as np
import os

logger = logging.getLogger(__name__)


class Movement_Syn_Optimizer(MetaOptimizer):
    """
    Implements a novel group pruning approach inspired by
    Victor Sanh, et. al.: Movement Pruning: Adaptive Sparsity by Fine-Tuning
    Such that the groups that have moved away most from 0 are kept,
    rest are pruned away.
    """
    mu=0    
    def __init__(
        self,
        config,
        #op_optimizer: torch.optim.Optimizer = torch.optim.SGD,# ProxSGD,    
        #op_optimizer: torch.optim.Optimizer = torch.optim.AdamW,# ProxSGD,    
        op_optimizer: torch.optim.Optimizer = ProxSGD,    
        op_optimizer_evaluate: torch.optim.Optimizer = torch.optim.SGD,     
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer

        Args:
            epochs (int): Number of epochs. Required for tau
            mu (float): corresponds to the Weight decay
            threshold (float): threshold of pruning
            op_optimizer (torch.optim.Optimizer: ProxSGD): optimizer for the op weights 
            op_optmizer_evaluate: (torch.optim.Optimizer): optimizer for the op weights            
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super(Movement_Syn_Optimizer, self).__init__()

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
        self.instantenous = config.search.instantenous
        self.mask = torch.nn.ParameterList()
        self.score = torch.nn.ParameterList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.warm_start_epochs = config.search.warm_start_epochs if hasattr(config.search, "warm_start_epochs") else 1
        self.k_initialized = False
        self.count_masking=0
        self.masking_interval = config.search.masking_interval
        num_classes = 10 if config.dataset=="cifar10" else 100
        self.num_classes = 120 if config.dataset=="ImageNet16-120" else num_classes
        self.k=1

        # MAKE SURE TO GIVE THE CORRENT PATH OF THE SCORES.PTH
        self.path=self.config.out_dir+'/'+self.config.search_space+'/'+self.config.dataset+'/'+self.config.optimizer+'/'+ str(self.config.seed)+'/search/scores.pth'
        
        """
        The further initializations are for future work, if we ever want to look into OOD robustness and distillation
        """        
        try:
            self.augmix = config.search.augmix
        except Exception:
            self.augmix = False
        try:
            self.no_jsd = config.search.no_jsd
        except Exception:
            self.no_jsd = False
        try:
            self.distill = config.search.distill            
        except Exception:
            self.distill = False
        if self.distill:
            import torchvision
            self.teacher = torchvision.models.resnet50()
            if self.dataset == "cifar10" or self.dataset == "cifar100":
                self.teacher.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, 
                                            kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
            try:
                teacher_path = config.search.teacher_path
            except Exception:
                #give path of the teacher model
                teacher_path = "/work/dlclarge2/agnihotr-ml/NASLib/naslib/data/augmix/cifar10_resnet50_model_best.pth.tar"
            teacher_state_dict = torch.load(teacher_path)['state_dict']
            new_teacher_state_dict={}
            for k, v in teacher_state_dict.items():
                k=k.replace("module.","")
                new_teacher_state_dict[k] = v
            self.teacher.load_state_dict(new_teacher_state_dict)
            self.teacher.to(device=self.device)
            self.teacher.eval()
            

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GSparse specific GSparseMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GMoveMixedOp(primitives))
    
    @staticmethod
    def add_alphas(edge):
        """
        Function to add the pruning flag 'mask' to the edges.
        And add a parameter 'weights' that will be used for storing the l2 norm
        of the weights of the operations which later is used for calculating
        the importance scores which is scored in "score" which are used to prune.

        """
        len_primitives = len(edge.data.op)
        mask = torch.nn.Parameter(
           torch.ones(size=[len_primitives], requires_grad=False), requires_grad=False
        )
        operations_to_mask = torch.nn.Parameter(torch.tensor(1), requires_grad=False
        )
        score = torch.nn.Parameter(
           torch.zeros(size=[len_primitives], requires_grad=False), requires_grad=False
        )
        weights = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[0.0]), requires_grad=False
        )
        dimension = torch.nn.Parameter(
           torch.FloatTensor(len_primitives*[0.0]), requires_grad=False
        )
        edge.data.set("mask", mask, shared=True)
        edge.data.set("operations_to_mask", operations_to_mask, shared=True)
        edge.data.set("score", score, shared=True)
        edge.data.set("weights", weights, shared=True)
        edge.data.set("dimension", dimension, shared=True)

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
                stdv = 1. / math.sqrt(edge.data.op[i].weight.size(0))
                edge.data.op[i].weight.data.uniform_(-stdv, stdv)                
    

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

        for mask in graph.get_all_edge_data("mask"):
            self.mask.append(mask) 
        for scores in graph.get_all_edge_data("score"):
            self.score.append(scores)    

        graph.parse()

        if self.op_optimizer == torch.optim.AdamW:            
            self.op_optimizer = self.op_optimizer(
                graph.parameters(),
                lr=0.0001,
                weight_decay=0.05,
            )
        elif self.op_optimizer == ProxSGD:            
            self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=self.mu,
            clip_bounds=(0,1),
            normalization=self.normalization,
            normalization_exponent=self.normalization_exponent
            )
        else:
            self.op_optimizer = self.op_optimizer(
                graph.parameters(),
                lr=self.config.search.learning_rate,
                momentum=self.config.search.momentum,
                weight_decay=self.mu,
            )
        max_iterations = (self.config.search.data_size/self.config.search.batch_size)*self.config.search.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.op_optimizer, max_iterations, verbose=False)
        graph.train()
        self.graph = graph
        self.scope = scope
    
    def jsd_loss(self, logits_train):
        logits_train, logits_aug1, logits_aug2 = torch.split(logits_train, len(logits_train) // 3)
        p_clean, p_aug1, p_aug2 = F.softmax(logits_train, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)

        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        augmix_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        return logits_train, augmix_loss

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

        if self.augmix:
            logits_train, augmix_loss = self.jsd_loss(logits_train)
        if self.distill:
            with torch.no_grad():
                logits_teacher = self.teacher(input_train)
                teacher_augmix_loss = 0
                if self.augmix:
                    logits_teacher, teacher_augmix_loss = self.jsd_loss(logits_teacher)
                teacher_loss = self.loss(logits_teacher, target_train) + teacher_augmix_loss


        train_loss = self.loss(logits_train, target_train)
        if self.augmix:
            train_loss = train_loss + augmix_loss
        if self.distill:
            train_loss = train_loss + teacher_loss

        train_loss.backward()#retain_graph=True)
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            self.graph.eval()
            logits_val = self.graph(input_val)
            if self.augmix:
                logits_val, _, _ = torch.split(logits_val, len(logits_val) // 3)
            val_loss = self.loss(logits_val, target_val)
        self.graph.train()
        
        return logits_train, logits_val, train_loss, val_loss

    def get_final_architecture(self):
        """
        Returns the final discretized architecture.

        Returns:
            Graph: The final architecture.
        """            
        all_scores=torch.load(self.path) #used the saved scores to get back the found architecture
        iter = iterator()                         

        def discretize_ops(edge):
            """
            The operation with the highest score is chosen as the edge operation.
            """
            if edge.data.has("score"):
                primitives = edge.data.op.get_embedded_ops()
                i =iter.get_data()
                scores = all_scores[i]
                edge.data.set("op", primitives[torch.argmax(torch.abs(scores)).item()])             

        graph = self.graph.clone().unparse()
        graph.prepare_discretization()               
        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)         
        
        graph.prepare_evaluation()
        graph.parse()
        graph.QUERYABLE=True
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
            return best_arch.query(Metric.VAL_ACCURACY, self.dataset)
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
        self.mask = self.mask.to(self.device)
        self.score = self.score.to(self.device)

    
    def new_epoch(self, epoch):
        """
        Just log the l2 norms of operation weights.
        """
        normalization_exponent=self.normalization_exponent
        instantenous = self.instantenous
        k_changed_this_epoch = False
        graph_weights=[]
        weights_iterator=0

        
        def update_l2_weights(edge):            
            """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of 
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.

            edge.data.weights is being used to accumulate scores over all groups
            edge.data.scores is then used to store the scores of a group of operations across cells
            """           
            nonlocal graph_weights
            nonlocal weights_iterator
            nonlocal epoch
            nonlocal loss_copy
            if epoch>0:
                import ipdb;ipdb.set_trace()
            if edge.data.has("score"):                
                weight=0
                for i in range(len(edge.data.op.primitives)):
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:                                
                                weight+= torch.sum(edge.data.op.primitives[i].op[j].weight.grad*graph_weights[weights_iterator]).to(device='cpu')
                                weights_iterator+=1
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                        weight+= torch.sum(edge.data.op.primitives[i].op[j].op[k].weight.grad*graph_weights[weights_iterator]).to(device='cpu')
                                        weights_iterator+=1
                                except AttributeError:
                                    continue                         
                        edge.data.weights[i]+=weight
                        weight=0.0                        
                        group_dim=torch.zeros(1)
                    except AttributeError:                           
                        size = 1
                        edge.data.weights[i]+=torch.sum(edge.data.op.primitives[i].weight.grad*graph_weights[weights_iterator]).to(device='cpu')
                        weights_iterator+=1
        
        def save_weights(edge):            
            nonlocal graph_weights
            if edge.data.has("score"):                
                weight=0.0                
                for i in range(len(edge.data.op.primitives)):
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:                                
                                graph_weights.append(torch.abs(edge.data.op.primitives[i].op[j].weight).to(device=self.device))
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                        graph_weights.append(torch.abs(edge.data.op.primitives[i].op[j].op[k].weight).to(device=self.device))
                                except AttributeError:
                                    continue                         
                        edge.data.weights[i]+=weight
                        weight=0.0                        
                    except AttributeError:                           
                        size = 1
                        graph_weights.append(torch.abs(edge.data.op.primitives[i].weight).to(device=self.device))
        
        def abs_weights(edge):            
            if edge.data.has("score"):                
                with torch.no_grad():
                    for i in range(len(edge.data.op.primitives)):
                        try:
                            for j in range(len(edge.data.op.primitives[i].op)):
                                try:                                
                                    edge.data.op.primitives[i].op[j].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].op[j].weight).to(device=self.device))
                                except (AttributeError, TypeError) as e:
                                    try:
                                        for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                            edge.data.op.primitives[i].op[j].op[k].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].op[j].op[k].weight).to(device=self.device))
                                    except AttributeError:
                                        continue                         
                        except AttributeError:                           
                            edge.data.op.primitives[i].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].weight).to(device=self.device))
        
        def calculate_scores(edge):
            if edge.data.has("score"):
                for i in range(len(edge.data.op.primitives)):
                    with torch.no_grad():
                        edge.data.score[i]+=edge.data.weights[i]#/edge.data.dimension[i]
                        
        # Currently not being used, just kept in case we plan to normalize scores.
        def normalize_scores(edge):
            if edge.data.has("score"):
                for i in range(len(edge.data.op.primitives)):
                    with torch.no_grad():
                        edge.data.score[i] = edge.data.score[i]#/len(count)                        

        k_initialized = self.k_initialized
        k = self.k
        scores_from_copy = []
        iterate_copy = 0
        ones = torch.ones([128,3,32,32]).cuda()

        def get_copy_scores(edge):
            nonlocal scores_from_copy
            if edge.data.has("score"):
                scores_from_copy.append(torch.clone(edge.data.score).detach().cpu())

        def masking(edge):
            nonlocal k_changed_this_epoch
            nonlocal epoch     
            nonlocal k_initialized
            nonlocal k
            nonlocal scores_from_copy
            nonlocal iterate_copy
            k_initialized = True
            if edge.data.has("score"):
                #scores = torch.clone(edge.data.score).detach().cpu()
                scores = scores_from_copy[iterate_copy]
                iterate_copy += 1
                edge.data.mask[torch.topk(torch.abs(scores), k=k, largest=False, sorted=False)[1]]=0                
                if k < len(edge.data.op.primitives)-1 and not k_changed_this_epoch:
                    k += 1                                       
                    k_changed_this_epoch = True                

        def reinitialize_scores(edge):
            if edge.data.has("score"):                
                for i in range(len(edge.data.weights)):                            
                    edge.data.score[i]=0

        def reinitialize_l2_weights(edge):
            if edge.data.has("score"):                
                for i in range(len(edge.data.weights)):                    
                    edge.data.weights[i]=0
                    edge.data.dimension[i]=0                    
                    if instantenous:
                        edge.data.score[i]=0

        if epoch >= 0:
            graph_copy = copy.deepcopy(self.graph)
            graph_copy.zero_grad()
            graph_copy.eval()
            graph_copy.update_edges(abs_weights, scope=self.scope, private_edge_data=True)
            graph_copy.update_edges(save_weights, scope=self.scope, private_edge_data=True)
            loss_copy = graph_copy(ones)
            logger.info('loss from ones: '+str(torch.sum(loss_copy)))
            torch.sum(loss_copy).backward()                
            graph_copy.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)
            graph_copy.update_edges(calculate_scores, scope=self.scope, private_edge_data=True)              
            #graph_copy.update_edges(get_copy_scores, scope=self.scope, private_edge_data=True)             
            weights_iterator=0
            graph_weights=[]
            #scores_from_copy = []
        if epoch >= self.warm_start_epochs and self.count_masking%self.masking_interval==0:
            #graph_copy = copy.deepcopy(self.graph)
            #graph_copy.zero_grad()
            #graph_copy.eval()
            #graph_copy.update_edges(abs_weights, scope=self.scope, private_edge_data=True)
            #graph_copy.update_edges(save_weights, scope=self.scope, private_edge_data=True)
            #loss_copy = graph_copy(ones)
            #torch.sum(loss_copy).backward()
            #graph_copy.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)
            #graph_copy.update_edges(calculate_scores, scope=self.scope, private_edge_data=True)              
            graph_copy.update_edges(get_copy_scores, scope=self.scope, private_edge_data=True)              
            self.graph.update_edges(masking, scope=self.scope, private_edge_data=True)
            self.k_initialized = k_initialized
            self.k = k           
            scores_from_copy = []
            iterate_copy = 0
            weights_iterator=0
            graph_weights=[]

        for score in graph_copy.get_all_edge_data("score"):                        
            with torch.no_grad():                
                self.score.append(score)
            
        weights_str = [
            ", ".join(["{:+.10f}".format(x) for x in a])
            + ", {}".format(np.argmax(torch.abs(a).detach().cpu().numpy()))
            for a in self.score
        ]
        logger.info(
            "Group scores (importance scores, last column max): \n{}".format(
                "\n".join(weights_str)
            )
        )
        
        """
        - The next few lines are used to store the scores after each epoch.
        - These scores are later used to recreate the found architecture.
        - This has to be done because pytorch and NASLib do not store gradients
          and out method is gradient's based.
        """
        all_scores=torch.nn.ParameterList()
        def save_score(edge):
            if edge.data.has("score"):
                with torch.no_grad():
                    all_scores.append(edge.data.score)

        graph_copy.update_edges(save_score, scope=self.scope, private_edge_data=True)
        mk_path = self.path.replace('/scores.pth','')
        os.makedirs(mk_path, exist_ok=True)
        torch.save(all_scores, self.path)
        all_scores=torch.nn.ParameterList()
        self.score = torch.nn.ParameterList()
        
        if epoch > 0:
            """
            The parameter weights is being used to calculate the scores, as a buffer.
            This buffer is reinitialized every epoch so that the scores are calculated correctly
            """
            self.graph.update_edges(reinitialize_l2_weights, scope=self.scope, private_edge_data=True)
            count = []

        if epoch >= self.warm_start_epochs and self.instantenous:
            """
            If the scores being used are instantenous i.e. if only the scores right before the
            masking steps are being considered for masking then the scores are reinitialized
            at the end of every epoch i.e. the scores are not accumulated over epochs.
            """
            self.graph.update_edges(reinitialize_scores, scope=self.scope, private_edge_data=True)
        
        if epoch >= self.warm_start_epochs and self.count_masking%self.masking_interval==0:
            """
            The scores are reinitialized after every masking step
            """
            self.mask = torch.nn.ParameterList()
            self.graph.update_edges(reinitialize_scores, scope=self.scope, private_edge_data=True)
                
            for mask in self.graph.get_all_edge_data("mask"):
                self.mask.append(mask)
        if epoch >= self.warm_start_epochs:
            self.count_masking += 1
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
            #"op_optimizer_evaluate": self.op_optimizer_evaluate,            
        }

class GMoveMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        - Initialize the mixed op for Group Movement.

        - Each output of each operation is multiplied with that
          operations weight and mask to get the actual output of
          that operation.

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)        
        self.min_cuda_memory = min_cuda_memory

    def get_weights(self, edge_data):
        return edge_data.mask
    
    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):            
        summed = 0  
        for mask, op in zip(weights, self.primitives):
            if hasattr(op, "op"):
                summed += mask * op(x, None)                
            else:
                summed += mask * op(x, None)*op.weight                   
        return F.normalize(summed) 

class iterator():
    """
    An iterator function, to iterate over the scores stored while loading them back in get_final_architecture()
    """
    def __init__(self, i=-1):
        self.i=i                
    def get_data(self):
        self.i+=1
        return self.i       
