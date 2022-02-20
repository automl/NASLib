
from torch._C import device
from naslib.optimizers.core.metaclasses import MetaOptimizer

import torch
import numpy as np
import logging

from naslib.optimizers.oneshot.configurable.components import AbstractArchitectureSampler, AbstractCombOpModifier, AbstractEdgeOpModifier

logger = logging.getLogger(__name__)

class ConfigurableOptimizer(MetaOptimizer):
    """ A configurable optimizer."""
    def __init__(
        self,
        config,
        arch_sampler: AbstractArchitectureSampler,
        mixed_op_modifier: AbstractEdgeOpModifier,
        comb_op_modifier: AbstractCombOpModifier,

        # The rest are the same as the other optimizers
        op_optimizer=torch.optim.SGD,
        arch_optimizer=torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss()
    ):
        super(ConfigurableOptimizer, self).__init__()
        self.config = config
        self.arch_sampler = arch_sampler
        self.mixed_op_modifier = mixed_op_modifier
        self.comb_op_modifier = comb_op_modifier

        self.op_optimizer = op_optimizer
        self.arch_optimizer = arch_optimizer
        self.loss = loss_criteria
        self.grad_clip = self.config.search.grad_clip

        self.architectural_weights = torch.nn.ParameterList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.arch_sampler.set_device(self.device)

    def _init_architectural_weights(self, graph):
        arch_modifiers = [self.arch_sampler, self.comb_op_modifier]
        arch_weights = [item.get_arch_weights(graph) for item in arch_modifiers if item.get_arch_weights(graph) is not None]
        arch_weights_flat = [alpha for weights in arch_weights for alpha in weights]
        print(arch_weights_flat)

        for param in arch_weights_flat:
            self.architectural_weights.append(param)

    def _init_optimizers(self, graph):
        if self.arch_optimizer is not None:
            self.arch_optimizer = self.arch_optimizer(
                self.architectural_weights.parameters(),
                lr=self.config.search.arch_learning_rate,
                betas=(0.5, 0.999),
                weight_decay=self.config.search.arch_weight_decay,
            )

        self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=self.config.search.weight_decay,
        )

    def _arch_optimizer_step(self, train, target):
        return self._optimizer_step(
            graph=self.graph,
            optimizer=self.arch_optimizer,
            loss_criterion=self.loss,
            data=train,
            target=target,
            grad_clip=self.grad_clip,
            is_arch_optimizer=True
        )

    def _graph_optimizer_step(self, train, target):
        return self._optimizer_step(
            graph=self.graph,
            optimizer=self.op_optimizer,
            loss_criterion=self.loss,
            data=train,
            target=target,
            grad_clip=self.grad_clip,
            is_arch_optimizer=False
        )

    def _optimizer_step(self, graph, optimizer, loss_criterion, data, target, grad_clip, is_arch_optimizer):
        optimizer.zero_grad()
        logits = graph(data)
        loss = loss_criterion(logits, target)
        loss.backward()

        if grad_clip:
            parameters = self.architectural_weights.parameters() if is_arch_optimizer else graph.parameters()
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)

        optimizer.step()

        return logits, loss

    def adapt_search_space(self, search_space, scope=None):
        self.search_space = search_space
        graph = search_space.clone()

        # If there is no scope defined, use the search space default one
        if not scope:
            scope = graph.OPTIMIZER_SCOPE

        self.arch_sampler.update_graph(graph, scope) # DARTS, DrNAS, or GDAS
        self.mixed_op_modifier.update_graph(graph, scope) # Partial Connections
        self.comb_op_modifier.update_graph(graph, scope) # Edge Normalization

        graph.parse()

        self._init_architectural_weights(graph)
        self._init_optimizers(graph)
        self.graph = graph
        self.scope = scope

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # Sample architecture weights
        self.arch_sampler.sample_arch_weights(self.graph, self.scope)

        # Update architecture weights
        logits_val, val_loss = self._arch_optimizer_step(input_val, target_val)

        # Sample architectural weights again
        # Ideally, save the RNG state from before the previous sampling and set it
        # again here
        self.arch_sampler.sample_arch_weights(self.graph, self.scope)

        # Update op weights
        logits_train, train_loss = self._graph_optimizer_step(input_train, target_train)

        # Remove the sampled architecture weights
        self.arch_sampler.remove_sampled_arch_weights(self.graph, self.scope)

        return logits_train, logits_val, train_loss, val_loss

    def new_epoch(self, epoch):
        self.arch_sampler.new_epoch()

    def get_op_optimizer(self):
        return self.op_optimizer.__class__

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
        self.graph = self.graph.to(self.device)
        self.architectural_weights = self.architectural_weights.to(self.device)

    def new_epoch(self, epoch):
        alpha_str = [
            ", ".join(["{:+.06f}".format(x) for x in a])
            + ", {}".format(np.argmax(a.detach().cpu().numpy()))
            for a in self.architectural_weights
        ]
        logger.info(
            "Arch weights (alphas, last column argmax): \n{}".format(
                "\n".join(alpha_str)
            )
        )

        self.arch_sampler.new_epoch()
        super().new_epoch(epoch)


    def get_final_architecture(self):
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                edge.data.set("op", primitives[np.argmax(alphas)])

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        graph = graph.to(self.device)
        return graph