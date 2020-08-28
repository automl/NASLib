import networkx as nx
import copy
import logging
import torch

from collections.abc import Iterable
from networkx.algorithms.dag import lexicographical_topological_sort
from torch.utils.tensorboard import SummaryWriter

from naslib.utils.utils import drop_path, cat_channels
from naslib.search_spaces.core.metaclasses import MetaGraph
from naslib.search_spaces.core.primitives import Identity, AbstractPrimitive

from naslib.utils import iter_flatten

# TODO: do logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeData():
    """
    Class that holds data for each edge.
    Data can be shared between instances of the graph
    where the edges lives in.

    Also defines the default key 'op', which is `Identity()`. It must
    be private always.

    Items can be accessed directly as attributes with `.key` or
    in a dict-like fashion with `[key]`. To set a new item use `.set()`.
    """

    def __init__(self, data={}):
        """
        Initializes a new EdgeData object.
        'op' is set as Identity() and private by default

        Args:
            data (dict): Inject some initial data. Will be always private.
        """
        self._private = {}
        self._shared = {}
        self.set('op', Identity(), shared=False)
        for k, v in data.items():
            self.set(k, v, shared=False)


    def has(self, key):
        """
        Checks whether `key` exists.

        Args:
            key (str): The key to check.
        
        Returns:
            bool: True if key exists, False otherwise.

        """
        if key in self._private.keys() or key in self._shared.keys():
            return True
        else:
            return False


    def __getitem__(self, key):
        return self.__getattr__(key)

    
    def __getattr__(self, name: str):
        if name.startswith("__"):       # Required for deepcoy, not sure why
            raise AttributeError(name)  # 
        if name in self._private:
            return self._private[name]
        elif name in self._shared:
            return self._shared[name]
        else:
            raise AttributeError("Cannot find field '{}' in the given EdgeData!".format(name))
    

    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise ValueError("not allowed. use set().")


    def __str__(self):
        return "private: <{}>, shared: <{}>".format(str(self._private), str(self._shared))


    def __repr__(self):
        return self.__str__()


    def update(self, data):
        """
        Update the data in here. If the data is added as dict,
        then all variables will be handled as private.

        Args:
            data (EdgeData or dict): If dict, then values will be set as
                private. If EdgeData then all entries will be replaced.
        """
        if isinstance(data, dict):
            for k, v in data.items():
                self.set(k, v)
        elif isinstance(data, EdgeData):
            # TODO: do update and not replace!
            self.__dict__.update(data.__dict__)
        else:
            raise ValueError("Unsupported type {}".format(data))


    def remove(self, key):
        """
        Removes an item from the EdgeData

        Args:
            key (str): The key for the item to be removed.
        """
        if key in self._private:
            del self._private[key]
        elif key in self._shared:
            del self._shared[key]
        else:
            raise KeyError("Tried to delete unkown key {}".format(key))


    def copy(self):
        """
        When a graph is copied to get multiple instances (e.g. when 
        reusing subgraphs at more than one location) then
        this function will be called for all edges.

        It will create a deep copy for the private entries but
        only a shallow copy for the shared entries. E.g. architectural
        weights should be shared, but parameters of a 3x3 convolution not.

        Therefore 'op' must be always private.

        Returns:
            EdgeData: A new EdgeData object with independent private
                items, but shallow shared items.
        """
        new_self = EdgeData()
        new_self._private = copy.deepcopy(self._private)
        new_self._shared = self._shared
        return new_self


    def set(self, key, value, shared=False):
        """
        Used to assign a new item to the EdgeData object.

        Args:
            key (str): The key.
            value (object): The value to store
            shared (bool): Default: False. Whether the item should
                be a shallow copy between different instances of EdgeData
                (and consequently between different instances of Graph).
        """
        assert isinstance(key, str), "Accepting only string keys, got {}".format(type(key))
        if shared:
            if key in self._private:
                raise ValueError("Key {} alredy defined as non-shared")
            else:
                self._shared[key] = value
        else:
            if key in self._shared:
                raise ValueError("Key {} alredy defined as shared")
            else:
                self._private[key] = value


    def clone(self):
        """
        Return a true deep copy of EdgeData. Even shared
        items are not shared anymore.

        Returns:
            EdgeData: New independent instance.
        """
        return copy.deepcopy(self)


class Graph(nx.DiGraph, torch.nn.Module):
    """
    Can sit on an edge or on a node.
    if sitting on edge, then must have exactly
    one input. if sitting on node it can have
    `k` inputs and `set_inputs` must be called.
    """

    OPTIMIZER_SCOPE = "all"

    
    def __init__(self):
        nx.DiGraph.__init__(self)
        torch.nn.Module.__init__(self)
        
        # Replace the default dicts at the edges with `EdgeData` objects
        # `EdgeData` can be easily customized and allow shared parameters
        # across different Graph instances.
        self.edge_attr_dict_factory = lambda: EdgeData()

        # Replace the default dicts at the nodes to include `input` from the beginning.
        # `input` is required for storing the results of incoming edges.
        self.node_attr_dict_factory = lambda: dict({'input': {}, 'comb_op': sum})

        self.name = None
        self.scope = None
        self.input_node_idxs = None


    def __eq__(self, other): 
        return self.name == other.name


    def __hash__(self):
        return hash(self.name)


    def set_scope(self, scope):
        self.scope = scope
        return self


    def set_input(self, node_idxs):
        """
        Route the input from specific parent edges to the input nodes of 
        this subgraph. Inputs are assigned in lexicographical order.

        Example: 
        - Parent node (i.e. node where `self` is located on) has two
          incoming edges from nodes 3 and 5.
        - `self` has two input nodes 1 and 2 (i.e. nodes without
          an incoming edge)
        - `node_idxs = [5, 3]`
        Then input of node 5 is routed to node 1 and input of node 3
        is routed to node 2.

        Similarly, if `node_idxs = [5, 5]` then input of node 5 is routed
        to both node 1 and 2. Warning: In this case the output of another 
        incoming edge is ignored!
        """
        num_innodes = sum([self.in_degree(n) == 0 for n in self.nodes])
        assert num_innodes == len(node_idxs), \
            "Expecting node index for every input node. Excpected {}, got {}".format(num_innodes, len(node_idxs))
        self.input_node_idxs = node_idxs
        return self


    def num_input_nodes(self):
        return sum(self.in_degree(n) == 0 for n in self.nodes)


    def _assign_x_to_nodes(self, x):
        """
        Assign x to the input nodes of self. Depending whether on
        edge or nodes.

        Performs also several sanity checks of the input.
        """
        # We need dict in case of cell and int in case of motif
        assert isinstance(x, dict) or isinstance(x, torch.Tensor)

        if self.input_node_idxs is None:
            assert self.num_input_nodes() == 1, "There are more than one input nodes but input indeces are not defined."
            assert len(list(self.predecessors(1))) == 0, "Expecting node 1 to be the parent."
            assert 'subgraph' not in self.nodes[1].keys(), "Expecting node 1 not to have a subgraph as it serves as input node."
            assert isinstance(x, torch.Tensor)
            self.nodes[1]['input'] = {0: x}
        else:
            # assign the input to the corresponding nodes
            assert all([i in x.keys() for i in self.input_node_idxs]), "got x from an unexpected input edge"
            if self.num_input_nodes() > len(x):
                # here is the case where the same input is assigned to more than one node
                # this can happen when there are cells with two inputs but at the very first
                # layer of the network, there is just one output (i.e. the data inputed to the
                # makro input node). Handle it and log a Info. This should happen only rarly
                logger.debug("We are using the same x for two inputs in graph {}".format(self.name))
            input_node_iterator = iter(self.input_node_idxs)
            for node_idx in nx.algorithms.dag.lexicographical_topological_sort(self):
                if self.in_degree(node_idx) == 0:
                    self.nodes[node_idx]['input'] = {0: x[next(input_node_iterator)]}


    def forward(self, x):
        """
        Forward some data through the graph. This is done recursively
        in case there are graphs defined on nodes or as 'op' on edges.
        """
        logger.debug("Graph {} called. Input {}.".format(self.name, x.shape if isinstance(x, torch.Tensor) else x))
        
        # Assign x to the corresponding input nodes
        self._assign_x_to_nodes(x)

        for node_idx in lexicographical_topological_sort(self):
            node = self.nodes[node_idx]
            logger.debug("Node {}-{}, current data {}, start processing...".format(self.name, node_idx, node))
            
            # node internal: process input if necessary
            if 'subgraph' in node:
                x = node['subgraph'].forward(node['input'])
            else:
                if len(node['input'].values()) == 1:
                    x = list(node['input'].values())[0]
                else:
                    x = node['comb_op'](list(node['input'].values()))
            
            # outgoing edges: process all outgoing edges
            for neigbor_idx in self.neighbors(node_idx):
                edge_data = self.get_edge_data(node_idx, neigbor_idx)
                # inject edge data only for AbstractPrimitive, not Graphs
                if isinstance(edge_data.op, Graph):
                    edge_output = edge_data.op.forward(x)
                elif isinstance(edge_data.op, AbstractPrimitive):
                    logger.debug("Processing op {}".format(edge_data.op))
                    edge_output = edge_data.op.forward(x, edge_data=edge_data)
                else:
                    raise ValueError("Unknown class as op: {}. Expected either Graph or AbstactPrimitive".format(
                            edge_data.op
                        ))
                self.nodes[neigbor_idx]['input'].update({node_idx: edge_output})
            
            logger.debug("Node {}-{}, processing done.".format(self.name, node_idx))

        logger.debug("Graph {} exiting. Output {}.".format(self.name, x))
        return x


    def parse(self):
        for node_idx in lexicographical_topological_sort(self):
            if 'subgraph' in self.nodes[node_idx]:
                self.nodes[node_idx]['subgraph'].parse()
                self.add_module("{}-subgraph_at({})".format(self.name, node_idx), self.nodes[node_idx]['subgraph'])
            for neigbor_idx in self.neighbors(node_idx):
                edge_data = self.get_edge_data(node_idx, neigbor_idx)
                if isinstance(edge_data.op, Graph):
                    edge_data.op.parse()
                self.add_module("{}-edge({},{})".format(self.name, node_idx, neigbor_idx), edge_data.op)


    def _get_child_graphs(self, single_instances=False):
        graphs = []
        for node_idx in lexicographical_topological_sort(self):
            node_data = self.nodes[node_idx]
            if 'subgraph' in node_data:
                graphs.append(node_data['subgraph'])
                graphs.append(node_data['subgraph']._get_child_graphs())

        for _, _, edge_data in self.edges.data():
            if isinstance(edge_data.op, Graph):
                graphs.append(edge_data.op)
                graphs.append(edge_data.op._get_child_graphs())
            elif isinstance(edge_data.op, list):
                for op in edge_data.op:
                    if isinstance(op, Graph):
                        graphs.append(op)
                        graphs.append(op._get_child_graphs())
            elif isinstance(edge_data.op, AbstractPrimitive):
                # maybe it is an embedded op?
                embedded_ops = edge_data.op.get_embedded_ops()
                if embedded_ops is not None:
                    assert isinstance(embedded_ops, list), "Unsupported return of `get_embedded_ops()` of {}. Expected list, got {}".format(edge_data.op, type(embedded_ops))
                    for child_op in edge_data.op.get_embedded_ops():
                        if isinstance(child_op, Graph):
                            graphs.append(child_op)
                            graphs.append(child_op._get_child_graphs())
            else:
                raise ValueError("Unknown format of op: {}".format(edge_data.op))
        
        graphs = [g for g in iter_flatten(graphs)]
        
        if single_instances:
            return sorted(list(set(graphs)), key=lambda x: x.name)
        else:
            return sorted(graphs, key=lambda x: x.name)


    def get_all_edge_data(self, key, scope='all', private_edge_data=False):
        result = []
        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or (graph.scope is not None and graph.scope in scope):
                for u, v, edge_data in graph.edges.data():
                    if edge_data.has(key):
                        result.append(edge_data[key])
        return result


    def update_edges(self, update_func, scope="all", private_edge_data=False):
        """
        This updates the graph and all child graphs, but only one
        instance of each.
        `update_func(current_edge_data)`. This way optimizers
        can initialize and store necessary information at edges.

        scope can be "all" or list of graph names to be updated.

        private_edge_data: if set to true, this means update_func will be
                applied to all edges. THIS IS NOT RECOMMENDED FOR SHARED 
                ATTRIBUTES. Shared attributes should be set only once, we
                take care it is syncronized across all copies of this graph.
                
                The only usecase for setting it to true is when actually changing
                `op` during the initialization of the optimizer (e.g. replacing it
                with MixedOp or SampleOp)
        """
        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or (graph.scope is not None and graph.scope in scope):
                print('updating {}'.format(graph.name))
                for u, v, edge_data in graph.edges.data():
                    graph.edges[u, v].update(update_func(current_edge_data=edge_data))


    def clone(self):
        """
        Deep copy of the current graph.

        Returns:
            Graph: Deep copy of the graph.
        """
        return copy.deepcopy(self)


    def reset_weights(self, inplace=False):
        """
        Resets the weights for the 'op' at all edges.

        Args:
            inplace (bool): Do the operation in place or
                return a modified copy.
        """
        if inplace:
            graph = self
        else:
            graph = self.clone()
        
        for m in self.modules():
            mm.reset_parameters()
        
        return graph


class GraphWrapper(Graph):
    """
    Provide methods currently required by optimizers/evaluators.

    TODO: Refactor optimizers so this is not needed anymode.
    """

    __name__ = "Cell"

    def get_node_op(self, n):
        if 'subgraph' in self.nodes[n]:
            return self.nodes[n]['subgraph']
        else:
            return None


####################################################################################
# TODO: Remove the parts below as they are the old search space implementation
#

class SampleOp(torch.nn.Module):
    
    def __init__(self, primitives):
        super(SampleOp, self).__init__()
        self.primitives = primitives if isinstance(primitives, list) else [primitives]
    
    def forward(self, x, edge_data):
        sample_state = edge_data.sample_state % len(self.primitives)
        print("using primitive {}".format(sample_state))
        return self.primitives[sample_state](x)
    
    def get_ops(self):
        return self.primitives


class EdgeOpGraph(nx.DiGraph, MetaGraph):
    """A graph whose edges contain operations"""

    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

    def save_graph(self):
        pass

    def parse(self, optimizer, *args, **kwargs):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]
            if 'preprocessing' in node_info:
                self.add_module('node' + str(node), node_info['preprocessing'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                for pred in preds:
                    # Replace the operation in the edge with an optimizer compatible one.
                    edge_data = self.get_edge_data(pred, node)
                    edge_data = optimizer.replace_function(edge_data, self)
                    self.add_module('edge(%d,%d)' % (pred, node), edge_data['op'])

    def forward(self, inputs):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo deal with multidigraph input.
        input_nodes = self.input_nodes()
        assert len(input_nodes) == len(inputs), "Number of inputs isn't the same as the number of inputs in the graph"

        for input_node, input in zip(input_nodes, inputs):
            self.nodes[input_node]['output'] = input

        for node in topo_order:
            node_info = self.nodes[node]
            if 'preprocessing' in node_info:
                node_info['output'] = node_info['preprocessing'](node_info['output'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                edge_outputs = []
                for pred in preds:
                    pred_info = self.nodes[pred]
                    assert 'output' in pred_info, 'Predecessor of current node has no output.'

                    # Evaluate the edge from the predecessor to the current node.
                    pred_output = pred_info['output']
                    edge_data = self.get_edge_data(pred, node)
                    edge_output = edge_data['op'](pred_output, **edge_data)

                    if hasattr(self, 'drop_path_prob'):
                        if self.training and self.drop_path_prob > 0:
                            edge_op = self.get_edge_op(pred, node)
                            if self.is_inter(node):
                                if len(edge_op) == 1:
                                    if not isinstance(edge_op._ops[0], Identity):
                                        edge_output = drop_path(edge_output,
                                                                self.drop_path_prob)
                                else:
                                    edge_output = drop_path(edge_output,
                                                            self.drop_path_prob)
                    edge_outputs.append(edge_output)

                # Combine evaluated input edges to form output of the cell
                comb_op = eval(node_info['comb_op'])
                self.nodes[node]['output'] = comb_op(edge_outputs)

        # Todo: Deal with multiple output EdgeOpGraphs
        return [self.nodes[node]['output'] for node in self.output_nodes()][0]


class NodeOpGraph(nx.DiGraph, MetaGraph):
    """A graph whose nodes contain operations"""

    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

    def save_graph(self, filename='graph.yaml', save_arch_weights=False):
        MetaGraph.save_graph(self, filename, save_arch_weights)

    def parse(self, optimizer, *args, **kwargs):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]
            if 'op' in node_info:
                self.add_module('node' + str(node), node_info['op'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                op = node_info['op']
                # Recursively run through EdgeOp graph cells
                if issubclass(type(op), EdgeOpGraph):
                    self.add_module('node' + str(node), op)
                    op.parse(optimizer)

    def forward(self, inputs):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo: Find better way to specify the input nodes
        self.nodes[0]['output'] = inputs
        for node in topo_order:
            node_info = self.nodes[node]

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                cell_input = [self.nodes[pred]['output'] for pred in preds]
                if 'transform' in node_info:
                    cell_input = node_info['transform'](cell_input)
                node_info['output'] = node_info['op'](cell_input)
        return [self.nodes[node]['output'] for node in self.output_nodes()][0]


if __name__ == '__main__':
    x = SimpleHierarchicalSpace()