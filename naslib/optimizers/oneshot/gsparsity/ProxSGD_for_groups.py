import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class ProxSGD(Optimizer):
    r"""Implements Prox-SGD algorithm for pruning 'groups'. 
    
    A group could be the same operation in all cells of the same type (normal or reduce). For example, one group is op 3 of edge 7 in Cells 0-1/3-4/5-7 (normal cells), another group is op 3 of edge 7 in Cell 2/5 (reduce cells).
    
    A group could also be the same operation in all cells in a stage (there may be multiple stages). For example, one group is op 3 of edge 7 in all cells of stage_normal_1 (Cells 0-1), another group is op 3 of edge 7 in all cells of stage_reduce_1 (Cell 2), another group is op 3 of edge 7 in all cells of stage_normal_2 (Cells 3-4), another group is op 3 of edge 7 in all cells of stage_reduce_2 (Cell 5), one group is op 3 of edge 7 in all cells of stage_normal_3 (Cells 6-7).
    """

    def __init__(self, params, lr=0.001, momentum=0.9,
                 weight_decay=None, clip_bounds=(None,None), normalization="none", normalization_exponent=0):
        if not 0.0 <= lr <= 1.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))
        if weight_decay is not None and weight_decay < 0:
            raise ValueError("Invalid weight decay parameter: {}".format(momentum))
        if not 0.0 <= normalization_exponent:
            raise ValueError("Invalid normalization exponent parameter: {}".format(momentum))

        self.normalization = normalization
        self.normalization_exponent = normalization_exponent
        defaults = dict(lr=lr, momentum=momentum, clip_bounds=clip_bounds, weight_decay=weight_decay)
        super(ProxSGD, self).__init__(params, defaults)      

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            b_group_norm = torch.zeros(1).cuda()
            if self.normalization == "none":
                dim_group = (torch.ones(1)).cuda()
            else: # "mul" or "div" normalization
                dim_group = (torch.zeros(1)).cuda()
                
            for x in group['params']:                
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['v_t'] = torch.zeros_like(x.data)

                v_t = state['v_t']

                state['step'] += 1

                momentum = group['momentum']
                # Decay the first and second moment running average coefficient
                v_t.mul_(1 - momentum).add_(momentum, grad)

                tau = 1
                b  = x - v_t / tau          

                if group['weight_decay'] is not None:
                    b_group_norm += torch.norm(b)**2
                    dim_group += torch.numel(x)

            b_group_norm = torch.sqrt(b_group_norm)
            
            #for x, x_is_scale_op in zip(group['params'], group['scale']):
            for x in group['params']:
                #print(len(x.size()),x.size()[0])
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                v_t = state['v_t'] # the momentum v_t is updated in the loop above
                lr = group['lr']
                
                tau = 1
                b  = x - v_t / tau          

                if group['weight_decay'] is not None: # operations are prunable
                    if self.normalization == "none": # no normalization
                        a  = group['weight_decay'] / tau
                    elif self.normalization == "mul": # normalize the weight decay by multiplying operation_dimension**exponent
                        a  = group['weight_decay'] * torch.pow(dim_group, self.normalization_exponent) / tau
                    elif self.normalization == "div": # normalize the weight decay by dividing operation_dimension**exponent
                        a  = group['weight_decay'] / torch.pow(dim_group, self.normalization_exponent) / tau
                        
                    x_hat = torch.clamp(1 - a / b_group_norm, min=0) * b
                    
#                     low, up = group['clip_bounds']
#                     if x_is_scale_op:
#                         if low is not None or up is not None:
#                             x_hat = x_hat.clamp_(low, up)
                else: #operations are nonprunable
                    x_hat = b

                x.data.add_(lr, x_hat - x)                        
        return loss