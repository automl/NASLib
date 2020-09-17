import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.optimizers.core.metaclasses import MetaOptimizer 
from naslib.optimizers.core.operations import MixedOp
from naslib.utils.utils import _concat, count_parameters_in_MB
import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DARTSOptimizer(MetaOptimizer):
    """
    Implementation of the DARTS paper as in 
        Liu et al. 2019: DARTS: Differentiable Architecture Search.
    """

    @staticmethod
    def add_alphas(current_edge_data):
        """
        Function to add the architectural weights to the edges.
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        len_primitives = len(current_edge_data.op)
        alpha = torch.nn.Parameter(1e-3 * torch.randn(size=[len_primitives], requires_grad=True))
        current_edge_data.set('alpha', alpha, shared=True)
        return current_edge_data


    @staticmethod
    def update_ops(current_edge_data):
        """
        Function to replace the primitive ops at the edges
        with the DARTS specific MixedOp.
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        primitives = current_edge_data.op
        current_edge_data.set('op', MixedOp(primitives))
        return current_edge_data


    def __init__(self, config,
            op_optimizer=torch.optim.SGD, 
            arch_optimizer=torch.optim.Adam, 
            loss_criteria=torch.nn.CrossEntropyLoss()
        ):
        # epochs, momentum, weight_decay, arch_learning_rate,
        # arch_weight_decay, grad_clip, *args, **kwargs
        """
        Initialize a new instance.

        Args:
            
        """
        super(DARTSOptimizer, self).__init__()
        
        self.config = config
        self.op_optimizer = op_optimizer
        self.arch_optimizer = arch_optimizer
        self.loss = loss_criteria
        self.grad_clip = self.config.grad_clip

        self.architectural_weights = torch.nn.ParameterList()

        self.perturb_alphas = None
        self.epsilon = 0


    def adapt_search_space(self, search_space, scope=None):
        # We are going to modify the search space
        graph = search_space.clone()

        # If there is no scope defined, let's use the search space default one
        if not scope:
            scope = graph.OPTIMIZER_SCOPE

        # 1. add alphas
        graph.update_edges(
            self.add_alphas,
            scope=scope,
            private_edge_data=False
        )

        # 2. replace primitives with mixed_op
        graph.update_edges(
            self.update_ops, 
            scope=scope,
            private_edge_data=True
        )

        for alpha in graph.get_all_edge_data('alpha'):
            self.architectural_weights.append(alpha)

        graph.parse()
        logger.info("Parsed graph:\n" + graph.modules_str())

        # Init optimizers
        self.arch_optimizer = self.arch_optimizer(
            self.architectural_weights.parameters(),
            lr=self.config.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=self.config.arch_weight_decay
        )

        self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        graph.train()
        
        self.graph = graph
        self.scope = scope
    
    def get_checkpointables(self):
        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "arch_optimizer": self.arch_optimizer,
            "arch_weights": self.architectural_weights,
        }


    def before_training(self):
        """
        Move the graph into cuda memory if available.
        """
        self.graph = self.graph.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.architectural_weights = self.architectural_weights.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    

    def new_epoch(self, epoch):
        """
        Just log the architecture weights.
        """
        logger.info("Arch weights: {}".format(([a for a in self.architectural_weights])))
        super().new_epoch(epoch)


    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val
        
        unrolled = False    # what it this?

        if unrolled:
            raise NotImplementedError()
        else:
            # Update architecture weights
            self.arch_optimizer.zero_grad()
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)
            val_loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.architectural_weights.parameters(), self.grad_clip)

            self.arch_optimizer.step()

            # Update op weights
            self.op_optimizer.zero_grad()
            logits_train = self.graph(input_train)
            train_loss = self.loss(logits_train, target_train)
            train_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
            self.op_optimizer.step()
        
        return logits_train, logits_val, train_loss, val_loss


    def get_final_architecture(self):
        logger.info("Arch weights before discretization: {}".format([a for a in self.architectural_weights]))
        graph = self.graph.unparse().clone()
        graph.prepare_discretization()

        def discretize_ops(current_edge_data):
            if current_edge_data.has('alpha'):
                primitives = current_edge_data.op.get_embedded_ops()
                alphas = current_edge_data.alpha.detach().cpu()
                current_edge_data.set('op', primitives[np.argmax(alphas)])
            return current_edge_data

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        return graph


    def get_op_optimizer(self):
        return self.op_optimizer.__class__


    def get_model_size(self):
        return count_parameters_in_MB(self.graph)


    def test_statistics(self):
        if self.graph.QUERYABLE:
            # record anytime performance
            best_arch = self.get_final_architecture()
            acc = best_arch.query('eval_acc1es', dataset=self.config.dataset, path=self.config.data)
            loss = best_arch.query('eval_losses', dataset=self.config.dataset, path=self.config.data)
            return acc, loss






    def _step(self, model, criterion, input_train, target_train, input_valid, target_valid, eta,
              network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(model, criterion, input_train, target_train,
                                         input_valid, target_valid, eta,
                                         network_optimizer)
        else:
            self._backward_step(model, criterion, input_valid, target_valid)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.architectural_weights.parameters(), self.grad_clip)
        self.optimizer.step()

    def _backward_step(self, model, criterion, input_valid, target_valid):
        """Compute 1st order approximation"""
        loss = self._loss(model, criterion, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, model, criterion, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(model, criterion, input_train, target_train, eta,
                                                      network_optimizer)
        unrolled_loss = self._loss(model=unrolled_model, criterion=criterion, input=input_valid,
                                   target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data if v.grad is not None else torch.zeros_like(v)
                  for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(model, criterion, vector, input_train, target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, model, criterion, input, target, eta, network_optimizer):
        loss = self._loss(model=model, criterion=criterion, input=input, target=target)
        theta = _concat(model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer'] for v in
                model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(
                loss, model.parameters()
            )
        ).data + self.network_weight_decay * theta

        unrolled_model = self._construct_model_from_theta(
            model, theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    def _construct_model_from_theta(self, model, theta):
        model_new = model.new()
        model_dict = model.state_dict()

        params, offset = {}, 0
        for k, v in model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, model, criterion, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)
        train_loss = self._loss(model, criterion, input=input, target=target)
        grads_p = torch.autograd.grad(train_loss, model.arch_parameters())

        for p, v in zip(model.parameters(), vector):
            p.data.sub_(2 * R, v)
        train_loss = self._loss(model, criterion, input=input, target=target)
        grads_n = torch.autograd.grad(train_loss, model.arch_parameters())

        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _loss(self, model, criterion, input, target):
        pred = model(input)
        return criterion(pred, target)




    # def replace_function(self, edge, graph):
    #     graph.architectural_weights = self.architectural_weights

    #     if 'op_choices' in edge:
    #         edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

    #         weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
    #             torch.nn.Parameter(1e-3 * torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

    #         self.architectural_weights[edge_key] = weights
    #         edge['arch_weight'] = self.architectural_weights[edge_key]
    #         edge['op'] = MixedOp(primitives=edge['op_choices'], **edge['op_kwargs'])

    #         if edge_key not in self.edges:
    #             self.edges[edge_key] = []
    #         self.edges[edge_key].append(edge)
    #     return edge

    # def forward_pass_adjustment(self, *args, **kwargs):
    #     if self.perturb_alphas is None:
    #         return

    #     for arch_key, arch_weight in self.architectural_weights.items():
    #         softmaxed_arch_weight = torch.nn.functional.softmax(arch_weight.clone(),
    #                                                             dim=-1)
    #         if self.perturb_alphas == 'random':
    #             perturbation = torch.zeros_like(softmaxed_arch_weight).uniform_(
    #                 -self.epsilon_alpha, self.epsilon_alpha
    #             )
    #             softmaxed_arch_weight.data.add_(perturbation)
    #             # clipping
    #             max_index = softmaxed_arch_weight.argmax()
    #             softmaxed_arch_weight.data.clamp_(0, 1)
    #             if softmaxed_arch_weight.sum() == 0.0:
    #                 softmaxed_arch_weight.data[max_index] = 1.0
    #             softmaxed_arch_weight.data.div_(softmaxed_arch_weight.sum())

    #         for edge in self.edges[arch_key]:
    #             edge['softmaxed_arch_weight'] = softmaxed_arch_weight
    #             edge['perturb_alphas'] = True

    # def undo_forward_pass_adjustment(self, *args, **kwargs):
    #     try:
    #         for arch_key in self.architectural_weights:
    #             for edge in self.edges[arch_key]:
    #                 del edge['softmaxed_arch_weight']
    #                 del edge['perturb_alphas']
    #     except KeyError:
    #         return

    # @classmethod
    # def from_config(cls, *args, **kwargs):
    #     nas_opt = cls(*args, **kwargs)
    #     return nas_opt

    # def new_epoch(self, epoch):
    #     if self.perturb_alphas is not None:
    #         self.epsilon_alpha = 0.03 + (self.epsilon - 0.03) * epoch/self.epochs

    # def add_perturbation(self, perturbation=None, epsilon=.3):
    #     if perturbation == None:
    #         return
    #     else:
    #         self.perturb_alphas = perturbation
