from naslib.optimizers.core.metaclasses import MetaOptimizer
import torch
import numpy as np
import os
import copy

class Syn(MetaOptimizer):
    def __init__(self, logger, epoch, object_self=None):                
        self.object_self=object_self
        self.object_self.logger=logger
        self.epoch=epoch
        super(Syn, self).__init__()
        
    def new_epoch(self, epoch):
        """
        Just log the l2 norms of operation weights.
        """        
        instantenous = self.object_self.instantenous
        logger=self.object_self.logger
        k_changed_this_epoch = False
        epoch=self.epoch
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
                                graph_weights.append(torch.abs(edge.data.op.primitives[i].op[j].weight).to(device=self.object_self.device))
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                        graph_weights.append(torch.abs(edge.data.op.primitives[i].op[j].op[k].weight).to(device=self.object_self.device))
                                except AttributeError:
                                    continue                         
                        edge.data.weights[i]+=weight
                        weight=0.0                        
                    except AttributeError:                           
                        size = 1
                        graph_weights.append(torch.abs(edge.data.op.primitives[i].weight).to(device=self.object_self.device))
        
        def abs_weights(edge):            
            if edge.data.has("score"):                
                with torch.no_grad():
                    for i in range(len(edge.data.op.primitives)):
                        try:
                            for j in range(len(edge.data.op.primitives[i].op)):
                                try:                                
                                    edge.data.op.primitives[i].op[j].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].op[j].weight).to(device=self.object_self.device))
                                except (AttributeError, TypeError) as e:
                                    try:
                                        for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                            edge.data.op.primitives[i].op[j].op[k].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].op[j].op[k].weight).to(device=self.object_self.device))
                                    except AttributeError:
                                        continue                         
                        except AttributeError:                           
                            edge.data.op.primitives[i].weight=torch.nn.Parameter(torch.abs(edge.data.op.primitives[i].weight).to(device=self.object_self.device))
        
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

        k_initialized = self.object_self.k_initialized
        k = self.object_self.k
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
            graph_copy = copy.deepcopy(self.object_self.graph)
            graph_copy.zero_grad()
            graph_copy.eval()
            graph_copy.update_edges(abs_weights, scope=self.object_self.scope, private_edge_data=True)
            graph_copy.update_edges(save_weights, scope=self.object_self.scope, private_edge_data=True)
            loss_copy = graph_copy(ones)
            logger.info('loss from ones: '+str(torch.sum(loss_copy)))
            torch.sum(loss_copy).backward()                
            graph_copy.update_edges(update_l2_weights, scope=self.object_self.scope, private_edge_data=True)
            graph_copy.update_edges(calculate_scores, scope=self.object_self.scope, private_edge_data=True)              
            #graph_copy.update_edges(get_copy_scores, scope=self.object_self.scope, private_edge_data=True)             
            weights_iterator=0
            graph_weights=[]
            #scores_from_copy = []
        if epoch >= self.object_self.warm_start_epochs and self.object_self.count_masking%self.object_self.masking_interval==0:
            #graph_copy = copy.deepcopy(self.object_self.graph)
            #graph_copy.zero_grad()
            #graph_copy.eval()
            #graph_copy.update_edges(abs_weights, scope=self.object_self.scope, private_edge_data=True)
            #graph_copy.update_edges(save_weights, scope=self.object_self.scope, private_edge_data=True)
            #loss_copy = graph_copy(ones)
            #torch.sum(loss_copy).backward()
            #graph_copy.update_edges(update_l2_weights, scope=self.object_self.scope, private_edge_data=True)
            #graph_copy.update_edges(calculate_scores, scope=self.object_self.scope, private_edge_data=True)              
            graph_copy.update_edges(get_copy_scores, scope=self.object_self.scope, private_edge_data=True)              
            self.object_self.graph.update_edges(masking, scope=self.object_self.scope, private_edge_data=True)
            self.object_self.k_initialized = k_initialized
            self.object_self.k = k           
            scores_from_copy = []
            iterate_copy = 0
            weights_iterator=0
            graph_weights=[]

        for score in graph_copy.get_all_edge_data("score"):                        
            with torch.no_grad():                
                self.object_self.score.append(score)
            
        weights_str = [
            ", ".join(["{:+.10f}".format(x) for x in a])
            + ", {}".format(np.argmax(torch.abs(a).detach().cpu().numpy()))
            for a in self.object_self.score
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

        graph_copy.update_edges(save_score, scope=self.object_self.scope, private_edge_data=True)
        mk_path = self.object_self.path.replace('/scores.pth','')
        os.makedirs(mk_path, exist_ok=True)
        torch.save(all_scores, self.object_self.path)
        all_scores=torch.nn.ParameterList()
        self.object_self.score = torch.nn.ParameterList()
        
        if epoch > 0:
            """
            The parameter weights is being used to calculate the scores, as a buffer.
            This buffer is reinitialized every epoch so that the scores are calculated correctly
            """
            self.object_self.graph.update_edges(reinitialize_l2_weights, scope=self.object_self.scope, private_edge_data=True)
            count = []

        if epoch >= self.object_self.warm_start_epochs and self.object_self.instantenous:
            """
            If the scores being used are instantenous i.e. if only the scores right before the
            masking steps are being considered for masking then the scores are reinitialized
            at the end of every epoch i.e. the scores are not accumulated over epochs.
            """
            self.object_self.graph.update_edges(reinitialize_scores, scope=self.object_self.scope, private_edge_data=True)
        
        if epoch >= self.object_self.warm_start_epochs and self.object_self.count_masking%self.object_self.masking_interval==0:
            """
            The scores are reinitialized after every masking step
            """
            self.object_self.mask = torch.nn.ParameterList()
            self.object_self.graph.update_edges(reinitialize_scores, scope=self.object_self.scope, private_edge_data=True)
                
            for mask in self.object_self.graph.get_all_edge_data("mask"):
                self.object_self.mask.append(mask)
        if epoch >= self.object_self.warm_start_epochs:
            self.object_self.count_masking += 1
        super().new_epoch(epoch)
        return self.object_self

    def adapt_search_space(self):
        return
    def get_final_architecture(self):
        return
    def get_op_optimizer(self):
        return