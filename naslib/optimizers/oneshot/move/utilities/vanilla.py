from naslib.optimizers.core.metaclasses import MetaOptimizer
import torch
import numpy as np
import os
from naslib.optimizers.oneshot.move.utilities.tools import Tools

class Vanilla(MetaOptimizer):
    def __init__(self, logger, epoch, object_self=None):                
        self.object_self=object_self
        #self.object_self.graph = object_self.val_graph if hasattr(self.object_self, 'val_graph') else object_self.graph
        self.logger=logger
        self.epoch=epoch
        self.tools = Tools(object_self=object_self, instantenous=object_self.instantenous)
        super(Vanilla, self).__init__()
        
    def new_epoch(self, epoch):
        """
        Just log the l2 norms of operation weights.
        """        
        instantenous = self.object_self.instantenous
        k_changed_this_epoch = False
        epoch=self.epoch

        k_initialized = self.object_self.k_initialized
        k = self.object_self.k      
          
        def masking(edge):
            nonlocal k_changed_this_epoch
            nonlocal epoch     
            nonlocal k_initialized
            nonlocal k
            k_initialized = True
            if edge.data.has("score"):
                scores = torch.clone(edge.data.score).detach().cpu()
                edge.data.mask[torch.topk(torch.abs(scores), k=k, largest=False, sorted=False)[1]]=0                
                if k < len(edge.data.op.primitives)-1 and not k_changed_this_epoch:
                    k += 1                                       
                    k_changed_this_epoch = True                

        if epoch > 0:
            self.object_self.graph.update_edges(self.tools.update_l2_weights, scope=self.object_self.scope, private_edge_data=True)
            self.object_self.graph.update_edges(self.tools.calculate_scores, scope=self.object_self.scope, private_edge_data=True)              
        if epoch >= self.object_self.warm_start_epochs and self.object_self.count_masking%self.object_self.masking_interval==0:
            self.object_self.graph.update_edges(masking, scope=self.object_self.scope, private_edge_data=True)
            self.object_self.k_initialized = k_initialized
            #self.object_self.k = self.object_self.k+1            
            self.object_self.k = k            

        self.object_self.score = torch.nn.ParameterList()
        for score in self.object_self.graph.get_all_edge_data("score"):                        
            with torch.no_grad():                
                self.object_self.score.append(score)
            
        weights_str = [
            ", ".join(["{:+.10f}".format(x) for x in a])
            + ", {}".format(np.argmax(torch.abs(a).detach().cpu().numpy()))
            for a in self.object_self.score
        ]
        self.logger.info(
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

        self.object_self.graph.update_edges(save_score, scope=self.object_self.scope, private_edge_data=True)
        mk_path = self.object_self.path.replace('/scores.pth','')
        os.makedirs(mk_path, exist_ok=True)
        torch.save(all_scores, self.object_self.path)
        all_scores=torch.nn.ParameterList()
        
        
        if epoch > 0:
            """
            The parameter weights is being used to calculate the scores, as a buffer.
            This buffer is reinitialized every epoch so that the scores are calculated correctly
            """
            self.object_self.graph.update_edges(self.tools.reinitialize_l2_weights, scope=self.object_self.scope, private_edge_data=True)
            count = []

        if epoch >= self.object_self.warm_start_epochs and self.object_self.instantenous:
            """
            If the scores being used are instantenous i.e. if only the scores right before the
            masking steps are being considered for masking then the scores are reinitialized
            at the end of every epoch i.e. the scores are not accumulated over epochs.
            """
            self.object_self.graph.update_edges(self.tools.reinitialize_scores, scope=self.object_self.scope, private_edge_data=True)
        
        if epoch >= self.object_self.warm_start_epochs and self.object_self.count_masking%self.object_self.masking_interval==0:
            """
            The scores are reinitialized after every masking step
            """
            self.object_self.mask = torch.nn.ParameterList()
            self.object_self.graph.update_edges(self.tools.reinitialize_scores, scope=self.object_self.scope, private_edge_data=True)
                
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
