import torch

class Tools():
    def __init__(self, object_self=None, instantenous=True):                        
        #global instantenous, masking_interval
        self.object_self=object_self
        self.instantenous = instantenous
        self.masking_interval = object_self.masking_interval
        super(Tools, self).__init__()

    @staticmethod
    def calculate_scores(edge):
        if edge.data.has("score"):
            for i in range(len(edge.data.op.primitives)):
                with torch.no_grad():
                    edge.data.score[i]+=edge.data.weights[i]#/edge.data.dimension[i]
    
    #@classmethod
    def normalize_scores(self, edge):        
        if edge.data.has("score"):
            for i in range(len(edge.data.op.primitives)):
                with torch.no_grad():
                    edge.data.score[i] = edge.data.score[i]/self.masking_interval                   
    @staticmethod
    def reinitialize_scores(edge):
        if edge.data.has("score"):                
            for i in range(len(edge.data.weights)):                            
                edge.data.score[i]=0
    
    #@classmethod
    def reinitialize_l2_weights(self, edge):                
        if edge.data.has("score"):                
            for i in range(len(edge.data.weights)):                    
                edge.data.weights[i]=0
                edge.data.dimension[i]=0                    
                if self.instantenous:
                    edge.data.score[i]=0      
    @staticmethod
    def save_score(edge):
        if edge.data.has("score"):
            with torch.no_grad():
                all_scores.append(edge.data.score)     

    @staticmethod
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
        if edge.data.has("score"):                
            weight=0.0                
            group_dim=torch.zeros(1)
            for i in range(len(edge.data.op.primitives)):
                try:
                    for j in range(len(edge.data.op.primitives[i].op)):
                        try:                                
                            group_dim += 1                                
                            weight+= torch.sum(edge.data.op.primitives[i].op[j].weight.grad*edge.data.op.primitives[i].op[j].weight).to(device=edge.data.weights.device)
                        except (AttributeError, TypeError) as e:
                            try:
                                for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                    group_dim += 1                                        
                                    weight+= torch.sum(edge.data.op.primitives[i].op[j].op[k].weight.grad*edge.data.op.primitives[i].op[j].op[k].weight).to(device=edge.data.weights.device)
                            except AttributeError:
                                continue                         
                    edge.data.weights[i]+=weight
                    edge.data.dimension[i]+=group_dim.item()                                                                         
                    weight=0.0                        
                    group_dim=torch.zeros(1)
                except AttributeError:                           
                    size = 1
                    edge.data.weights[i]+=torch.sum(edge.data.op.primitives[i].weight.grad*edge.data.op.primitives[i].weight).to(device=edge.data.weights.device)
                    edge.data.dimension[i]+=size 

    @staticmethod
    def not_abs_update_l2_weights(edge):            
        """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of 
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.

            edge.data.weights is being used to accumulate scores over all groups
            edge.data.scores is then used to store the scores of a group of operations across cells
        """            
        if edge.data.has("score"):                
            weight=0.0                
            group_dim=torch.zeros(1)
            for i in range(len(edge.data.op.primitives)):
                try:
                    for j in range(len(edge.data.op.primitives[i].op)):
                        try:                                
                            group_dim += 1                                
                            weight-= torch.sum(edge.data.op.primitives[i].op[j].weight.grad*edge.data.op.primitives[i].op[j].weight).to(device=edge.data.weights.device)
                        except (AttributeError, TypeError) as e:
                            try:
                                for k in range(len(edge.data.op.primitives[i].op[j].op)):                                        
                                    group_dim += 1                                        
                                    weight-= torch.sum(edge.data.op.primitives[i].op[j].op[k].weight.grad*edge.data.op.primitives[i].op[j].op[k].weight).to(device=edge.data.weights.device)
                            except AttributeError:
                                continue                         
                    edge.data.weights[i]+=weight
                    edge.data.dimension[i]+=group_dim.item()                                                                         
                    weight=0.0                        
                    group_dim=torch.zeros(1)
                except AttributeError:                           
                    size = 1
                    edge.data.weights[i]-=torch.sum(edge.data.op.primitives[i].weight.grad*edge.data.op.primitives[i].weight).to(device=edge.data.weights.device)
                    edge.data.dimension[i]+=size      