from abc import ABCMeta, abstractmethod
from enum import Enum, auto
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from naslib.optimizers.oneshot.darts.optimizer import DARTSMixedOp
from naslib.optimizers.oneshot.drnas.optimizer import DrNASMixedOp
from naslib.optimizers.oneshot.gdas.optimizer import GDASMixedOp
from naslib.search_spaces.core.primitives import EdgeNormalizationCombOp, MixedOp, PartialConnectionOp
from naslib.utils.utils import iter_flatten, AttrDict
class OptimizationStrategy(Enum):
    ALTERNATING = auto() # Arch weights optimized using validation data, model weights using train data
    SIMULTANEOUS = auto() # Arch and model weights optimized simultaneously using train data (e.g., SNAS)

class OneShotMixedOp(MixedOp):

    def __init__(self, primitives):
        super().__init__(primitives)

    def get_weights(self, edge_data):
        return edge_data.alpha

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        return sum(w * op(x, None) for w, op in zip(weights, self.primitives))


class SNASMixedOp(MixedOp):
    def __init__(self, primitives):
        super().__init__(primitives)

    def get_weights(self, edge_data):
        return edge_data.sampled_arch_weight

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights,edge_data):
        return sum(w * op(x, edge_data) for w, op in zip(weights, self.primitives))


class AbstractGraphModifier(metaclass=ABCMeta):
    @abstractmethod
    def update_graph_edges(self, graph, scope):
        raise NotImplementedError()

    #@abstractmethod
    def update_graph_nodes(self, graph, scope):
        raise NotImplementedError()

    def update_graph(self, graph, scope):
        self.update_graph_edges(graph, scope)
        self.update_graph_nodes(graph, scope)

    def new_epoch(self):
        pass


class AbstractEdgeOpModifier(AbstractGraphModifier):
    pass


class AbstractCombOpModifier(AbstractGraphModifier):
    @abstractmethod
    def get_arch_weights(self, graph):
        raise NotImplementedError


class NoEdgeOpModifer(AbstractEdgeOpModifier):
    def update_graph_edges(self, graph, scope):
        pass

    def update_graph_nodes(self, graph, scope):
        pass


class NoCombOpModifier(AbstractCombOpModifier):
    def update_graph_edges(self, graph, scope):
        pass

    def update_graph_nodes(self, graph, scope):
        pass

    def get_arch_weights(self, graph):
        return None


class EdgeNormalization(AbstractCombOpModifier):
    arch_weights_name = 'edge_normalization_beta'

    def _add_betas(self, edge):
        """
        Function to add the architectural weights to the edges.
        """
        beta = torch.nn.Parameter(
            1e-3 * torch.randn(size=[1], requires_grad=True)
        )
        edge.data.set(self.arch_weights_name, beta, shared=True)

    def _add_normalization_op(self, node, in_edges, out_edges):
        node_data = node[1]

        all_in_edges_final = True

        for _, edge_data in in_edges:
            if not edge_data.is_final():
                all_in_edges_final = False
                break

        if not in_edges or all_in_edges_final:
            return

        node_data['comb_op'] = EdgeNormalizationCombOp(node_data['comb_op'])

    def update_graph_edges(self, graph, scope):
        graph.update_edges(self._add_betas, scope=scope, private_edge_data=False)

    def update_graph_nodes(self, graph, scope):
        graph.update_nodes(self._add_normalization_op, scope=scope, single_instances=False)

    def get_arch_weights(self, graph):
        arch_weights = []
        for weight in graph.get_all_edge_data(self.arch_weights_name):
            arch_weights.append(weight)

        return arch_weights


class AbstractArchitectureSampler(AbstractGraphModifier):

    def __init__(self, arch_weights_modifier=None):
        self.arch_weights_modifier = arch_weights_modifier

    @abstractmethod
    def sample_arch_weights(self, graph, scope):
        raise NotImplementedError()

    @abstractmethod
    def remove_sampled_arch_weights(self, graph, scope):
        raise NotImplementedError()

    @abstractmethod
    def get_arch_weights(self, graph):
        raise NotImplementedError

    def weights_modifier_step(self, graph, scope):
        if self.arch_weights_modifier:
            self.arch_weights_modifier.step(graph, scope)

    def _update_ops(self, edge):
        primitives = edge.data.op
        op = self.__class__.mixed_op(primitives)
        edge.data.set('op', op)

        if self.arch_weights_modifier:
            self.arch_weights_modifier.register(op)

    def set_device(self, device):
        self.device = device

        if self.arch_weights_modifier is not None:
            self.arch_weights_modifier.set_device(device)

    def new_epoch(self):
        if self.arch_weights_modifier is not None:
            self.arch_weights_modifier.new_epoch()

class DARTSSampler(AbstractArchitectureSampler):
    mixed_op = DARTSMixedOp
    arch_weights_name = 'alpha'
    optimization_stratgey = OptimizationStrategy.ALTERNATING
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups={}

    def update_graph_edges(self, graph, scope):
        for u,v, edge_data in graph.edges.data():
            if not edge_data.is_final():
             edge = AttrDict(head=u, tail=v, data=edge_data)
             #print(edge)
             group = edge.data.group
             if group not in self.groups.keys():
                if group!="combi":
                    self.add_group_alphas(group,len(edge.data.op))  
                else:
                    self.add_group_alphas("ratio",3)      

        # 1. add alphas
    
        for u,v, edge_data in graph.edges.data():
            if not edge_data.is_final():
             edge = AttrDict(head=u, tail=v, data=edge_data)
             self.add_alphas(edge)
        #graph.update_edges(
        #    self.__class__.add_alphas, scope=scope, private_edge_data=False
        #)
        for u,v, edge_data in graph.edges.data():
            if not edge_data.is_final():
             edge = AttrDict(head=u, tail=v, data=edge_data)
             edge.data.set("discretize", False, shared=True)
        graph.update_edges(self._update_ops, scope=scope, private_edge_data=True)

    def get_arch_weights(self, graph):
        arch_weights = []
        for weight in graph.get_all_edge_data(self.arch_weights_name):
            arch_weights.append(weight)

        return arch_weights
    def add_group_alphas(self,group_name,len_primitives):
        alpha = torch.nn.Parameter(
            1e-3 * torch.randn(size=[len_primitives], requires_grad=True)
        )
        self.groups[group_name] = alpha #.cuda()
    def add_alphas(self,edge):
        """
        Function to add the architectural weights to the edges.
        """
        #print("Group",edge.data)
        #print("Group",edge.data.op)
        group = edge.data.group
        if group in self.groups.keys():
            edge.data.set("alpha", self.groups[group], shared=True)
        elif group == "combi":
            #alpha = torch.Tensor([x*y for x in torch.softmax(self.groups[edge.data.p1],dim=-1) for y in torch.softmax(self.groups[edge.data.p2],dim=-1)])
            #alpha = alpha.to("cuda")
            #print(alpha)
            edge.data.set("alpha_p1", self.groups[edge.data.p1] , shared=True)
            edge.data.set("alpha_p2", self.groups[edge.data.p2] , shared=True)
        else:
            len_primitives = len(edge.data.op)
            alpha = torch.nn.Parameter(1e-3 * torch.randn(size=[len_primitives], requires_grad=True))
            edge.data.set("alpha", alpha, shared=True)

    def update_graph_nodes(self, graph, scope):
        pass

    def sample_arch_weights(self, graph, scope):
        pass

    def remove_sampled_arch_weights(self, graph, scope):
        pass


class DrNASSampler(DARTSSampler):

    mixed_op = DrNASMixedOp

    def update_graph(self, graph, scope):
        super().update_graph(graph, scope)
        self.anchor = Dirichlet(
            torch.ones_like(
                torch.nn.utils.parameters_to_vector(self.get_arch_weights(graph))
            ).to(self.device)
        )

    def sample_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._sample_arch_weights(edge),
            scope=scope,
            private_edge_data=False,
        )

    def remove_sampled_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=self._remove_sampled_alphas,
            scope=scope,
            private_edge_data=False,
        )

    def _sample_arch_weights(self, edge):
        beta = F.elu(edge.data.get(self.arch_weights_name, None)) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        edge.data.set("sampled_arch_weight", weights, shared=True)

    def _remove_sampled_alphas(self, edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")


class GDASSampler(DARTSSampler):

    mixed_op = GDASMixedOp

    def __init__(self, epochs, tau_max, tau_min, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.tau_max = tau_max
        self.tau_min = tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = torch.Tensor([self.tau_max])

    def new_epoch(self):
        super().new_epoch()
        self.tau_curr += self.tau_step

    def update_graph(self, graph, scope):
        super().update_graph(graph, scope)
        graph.register_buffer('tau', self.tau_curr)

    def sample_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._sample_arch_weights(edge),
            scope=scope,
            private_edge_data=False,
        )

    def remove_sampled_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=self._remove_sampled_alphas,
            scope=scope,
            private_edge_data=False,
        )

    def _sample_arch_weights(self, edge):
        arch_parameters = torch.unsqueeze(edge.data.get(self.arch_weights_name, None), dim=0)

        while True:
            gumbels = -torch.empty_like(arch_parameters).exponential_().log()
            gumbels = gumbels.to(self.device)
            self.tau_curr = self.tau_curr.to(self.device)
            arch_parameters = arch_parameters.to(self.device)
            logits = (arch_parameters.log_softmax(dim=1) + gumbels) / self.tau_curr
            probs = torch.nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        weights = hardwts[0]

        edge.data.set("sampled_arch_weight", weights, shared=True)

    def _remove_sampled_alphas(self, edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")


class SNASSampler(DARTSSampler):



    def __init__(self, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.mixed_op = SNASMixedOp
        self.optimization_strategy = OptimizationStrategy.SIMULTANEOUS
        self.groups_sampled={}
    def new_epoch(self):
        super().new_epoch()

    def sample_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._sample_arch_weights(edge),
            scope=scope,
            private_edge_data=False,
        )

    def _sample_arch_weights(self, edge):
        if edge.data.group=="combi":
            log_alpha = torch.unsqueeze(torch.Tensor([x*y for x in torch.softmax(edge.data.alpha_p1,dim=-1) for y in torch.softmax(edge.data.alpha_p2,dim=-1)]), dim =0)
        else:
            log_alpha = edge.data.alpha
        
        if edge.data.group in self.groups_sampled.keys():
           u = self.groups_sampled[edge.data.group]
        elif edge.data.group=="combi":
            if edge.data.p1+edge.data.p2 in  self.groups_sampled.keys():
                u = self.groups_sampled[edge.data.p1+edge.data.p2]
            else:
                u = torch.zeros_like(log_alpha).uniform_()
                self.groups_sampled[edge.data.p1+edge.data.p2]=u
        else:
            u = torch.zeros_like(log_alpha).uniform_()
            self.groups_sampled[edge.data.group]=u
        softmax = torch.nn.Softmax(-1)
        weight = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)

        edge.data.set('sampled_arch_weight', weight, shared=True)

    def remove_sampled_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=self._remove_sampled_weights,
            scope=scope,
            private_edge_data=False,
        )

    def _remove_sampled_weights(self, edge):
        if edge.data.has('sampled_arch_weight'):
            edge.data.remove('sampled_arch_weight')


class PartialChannelConnection(AbstractEdgeOpModifier):
    def __init__(self, k):
        super(AbstractEdgeOpModifier, self).__init__()
        self.k = k

    def update_graph_edges(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._wrap_mixed_op(edge),
            scope=scope,
            private_edge_data=True,
        )
    def update_graph_nodes(self, graph, scope):
        pass
    def _wrap_mixed_op(self, edge):
        """Function to wrap PartialConnectionOps around the mixedops at the edges."""
        mixedop = edge.data.op

        assert isinstance(mixedop, MixedOp)
        edge.data.set("op", PartialConnectionOp(mixedop, k=self.k))


class DummyAbstractArchSampler(AbstractArchitectureSampler):
    def update_graph_nodes(self, graph, scope):
        pass

    def update_graph_edges(self, graph, scope):
        pass

    def get_arch_weights(self, graph):
        pass

    def sample_arch_weights(self, graph, scope):
        pass

    def remove_sampled_arch_weights(self, graph, scope):
        pass

    def new_epoch(self):
        pass

class AbstractMixedOpWeightsModifier:

    @abstractmethod
    def pre_process_fn(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def post_process_fn(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod
    def new_epoch(self):
        raise NotImplementedError()

    def set_device(self, device):
        self.device = device

    def register(self, mixed_op):
        mixed_op.set_pre_process_hook(self.pre_process_fn)
        mixed_op.set_post_process_hook(self.post_process_fn)


class RandomWeightPertubations(AbstractMixedOpWeightsModifier):
    perturbations_name = 'random_perturbation'

    def __init__(self, epsilon, epochs):
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epochs_max = epochs
        self.epoch = -1

    def step(self, graph, scope):
        graph.update_edges(self._sample_perturbations, scope=scope, private_edge_data=False)

    def new_epoch(self):
        self.epoch += 1
        self.epsilon = 0.03 + (self.epsilon_max - 0.03)*self.epoch/self.epochs_max

    def reset(self): #TODO: Is this needed?
        self.epoch = -1
        self.epsilon = self.epsilon_max

    def post_process_fn(self, weights, edge_data):
        return weights

    def pre_process_fn(self, weights, edge_data):
        perturbations = edge_data.get(self.perturbations_name, None).to(device=self.device)
        return weights.data.add(perturbations)

    def _sample_perturbations(self, edge):
        n_ops = len(edge.data.op.primitives)
        perturbations = torch.zeros([n_ops]).uniform_(-self.epsilon, self.epsilon)
        edge.data.set(self.perturbations_name, perturbations, shared=True)
