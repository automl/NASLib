import networkx as nx
import copy
import logging
import torch

from collections.abc import Iterable
from networkx.algorithms.dag import lexicographical_topological_sort
from torch.utils.tensorboard import SummaryWriter

from naslib.utils.utils import drop_path, cat_channels
from naslib.utils.logging import log_formats, log_first_n
from naslib.search_spaces.core.metaclasses import MetaGraph
from naslib.search_spaces.core.primitives import Identity, AbstractPrimitive

from naslib.utils import iter_flatten

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
    Base class for defining a search space. Add nodes and edges
    as for a directed acyclic graph in `networkx`. Nodes can contain
    graphs as children, also edges can contain graphs as operations.

    Note, if a graph is copied, the shared attributes of its edges are
    shallow copies whereas the private attributes are deep copies.

    To differentiate copies of the same graph you can define a `scope`
    with `set_scope()`. 

    **Graph at nodes:**
    >>> graph = Graph()
    >>> graph.add_node(1, subgraph=Graph())

    If the node has more than one input use `set_input()` to define the
    routing to the input nodes of the subgraph.

    **Graph at edges:**
    >>> graph = Graph()
    >>> graph.add_nodes_from([1, 2])
    >>> graph.add_edge(1, 2, EdgeData({'op': Graph()}))

    **Modify the graph after definition**

    If you want to modify the graph e.g. in an optimizer once
    it has been defined already use the function `update_edges()`. 
    
    """

    """
    Usually the optimizer does not operate on the whole graph, e.g. preprocessing
    and post-processing are excluded. Scope can be used to define that or to
    differentate instances of the "same" graph.
    """
    OPTIMIZER_SCOPE = "all"

    
    def __init__(self):
        """
        Initialise a graph. The edges are automatically filled with an EdgeData object
        which defines the default operation as Identity. The default combination operation
        is set as sum.
        """
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


    def __hash__(self):
        """
        As it is very complicated to compare graphs (i.e. check all edge
        attributes, do the have shared attributes, ...) use just the name
        for comparison.

        This is used when determining whether two instances are copies.
        """
        return hash(self.name)


    def set_scope(self, scope: str):
        """
        Sets the scope of this instance of the graph.

        The function should be used in a builder-like pattern
        `'subgraph'=Graph().set_scope("scope")`.

        Args:
            scope (str): the scope
        
        Returns:
            Graph: self with the setted scope.
        """
        self.scope = scope
        return self


    def add_node(self, node_index, **attr):
        """
        Adds a node to the graph.

        Note that adding a node using an index that has been used already
        will override its attributes.

        Args:
            node_index (int): The index for the node. Expect to be >= 1.
            **attr: The attributes which can be added in a dict like form.
        """
        assert node_index >= 1, "Expecting the node index to be greater or equal 1"
        super().add_node(node_index, **attr)


    def set_input(self, node_idxs: list):
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

        Should be used in a builder-like pattern: `'subgraph'=Graph().set_input([5, 3])`

        Args:
            node_idx (list): The index of the nodes where the data is coming from.
        
        Returns:
            Graph: self with input node indices set.

        """
        num_innodes = sum([self.in_degree(n) == 0 for n in self.nodes])
        assert num_innodes == len(node_idxs), \
            "Expecting node index for every input node. Excpected {}, got {}".format(num_innodes, len(node_idxs))
        self.input_node_idxs = node_idxs
        return self


    def num_input_nodes(self) -> int:
        """
        The number of input nodes, i.e. the nodes without an
        incoming edge.

        Returns:
            int: Number of input nodes.
        """
        return sum(self.in_degree(n) == 0 for n in self.nodes)


    def _assign_x_to_nodes(self, x):
        """
        Assign x to the input nodes of self. Depending whether on
        edge or nodes.

        Performs also several sanity checks of the input.

        Args:
            x (Tensor or dict): Input to be assigned.
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
        logger.debug("Graph {} called. Input {}.".format(self.name, log_formats(x)))
        
        # Assign x to the corresponding input nodes
        self._assign_x_to_nodes(x)

        for node_idx in lexicographical_topological_sort(self):
            node = self.nodes[node_idx]
            logger.debug("Node {}-{}, current data {}, start processing...".format(self.name, node_idx, log_formats(node)))
            
            # node internal: process input if necessary
            if ('subgraph' in node and 'comb_op' not in node) or ('comb_op' in node and 'subgraph' not in node):
                log_first_n(logging.WARN, "Comb_op is ignored if subgraph is defined!", n=1)
            # TODO: merge 'subgraph' and 'comb_op'. It is basicallly the same thing. Also in parse()
            if 'subgraph' in node:
                x = node['subgraph'].forward(node['input'])
            else:
                if len(node['input'].values()) == 1:
                    x = list(node['input'].values())[0]
                else:
                    x = node['comb_op']([node['input'][k] for k in sorted(node['input'].keys())])
            
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

        logger.debug("Graph {} exiting. Output {}.".format(self.name, log_formats(x)))
        return x


    def parse(self):
        """
        Convert the graph into a neural network which can then
        be optimized by pytorch.
        """
        for node_idx in lexicographical_topological_sort(self):
            if 'subgraph' in self.nodes[node_idx]:
                self.nodes[node_idx]['subgraph'].parse()
                self.add_module("{}-subgraph_at({})".format(self.name, node_idx), self.nodes[node_idx]['subgraph'])
            else:
                if isinstance(self.nodes[node_idx]['comb_op'], torch.nn.Module):
                    self.add_module("{}-comb_op_at({})".format(self.name, node_idx), self.nodes[node_idx]['comb_op'])
            for neigbor_idx in self.neighbors(node_idx):
                edge_data = self.get_edge_data(node_idx, neigbor_idx)
                if isinstance(edge_data.op, Graph):
                    edge_data.op.parse()
                self.add_module("{}-edge({},{})".format(self.name, node_idx, neigbor_idx), edge_data.op)


    def _get_child_graphs(self, single_instances: bool = False) -> list:
        """
        Get all child graphs of the current graph.

        Args:
            single_instances (bool): Whether to return multiple instances
                (i.e. copies) of the same graph. When changing shared data
                this should be set to True.
        
        Returns:
            list: A list of all child graphs (can be empty)
        """
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


    def get_all_edge_data(self, key: str, scope='all', private_edge_data: bool = False) -> list:
        """
        Get edge attributes of this graph and all child graphs in one go.

        Args:
            key (str): The key of the attribute
            scope (str): The scope to be applied
            private_edge_data (bool): Whether to return data from graph copies as well.
        
        Returns:
            list: All data in a list.
        """
        result = []
        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or (graph.scope is not None and graph.scope in scope):
                for u, v, edge_data in graph.edges.data():
                    if edge_data.has(key):
                        result.append(edge_data[key])
        return result


    def _verify_update_function(update_func: callable, private_edge_data: bool):
        """
        Verify that the update function actually modifies only
        shared/private edge data attributes based on setting of
        `private_edge_data`.

        Args:
            update_func (callable): callable that expects one argument
                named `current_edge_data`.
            private_edge_data (bool): Whether the update function is applied
                to all graph instances including copies or just to one instance
                per graph
        """

        test = EdgeData()
        test.set('shared', True, shared=True)
        test.set('op', True)

        try:
            result = update_func(current_edge_data=test.clone())
        except:
            logger.warn("Update function could not be veryfied. Be cautious with the "
                "setting of `private_edge_data` in `update_edges()`")
            return

        assert isinstance(result, EdgeData), "Update function does not return the edge data object."

        if private_edge_data:
            assert result._shared == test._shared, \
                "The update function changes shared data although `private_edge_data` set to True. " \
                "This is not the indended use of `update_edges`. The update function should only modify " \
                "private edge data."
        else:
            assert result._private == test._private, \
                "The update function changes private data although `private_edge_data` set to False. " \
                "This is not the indended use of `update_edges`. The update function should only modify " \
                "shared edge data."


    def update_edges(self, update_func: callable, scope="all", private_edge_data: bool = False):
        """
        This updates the edge data of this graph and all child graphs.
        This is the preferred way to manipulate the edges after the definition
        of the graph, e.g. by optimizers who want to insert their own op. 
        `update_func(current_edge_data)`. This way optimizers
        can initialize and store necessary information at edges.

        Args:
            update_func (callable): Function which accepts one argument called `current_edge_data`.
                and returns the modified EdgeData object.
            scope (str or list(str)): Can be "all" or list of scopes to be updated.
            private_edge_data (bool): If set to true, this means update_func will be
                applied to all edges. THIS IS NOT RECOMMENDED FOR SHARED 
                ATTRIBUTES. Shared attributes should be set only once, we
                take care it is syncronized across all copies of this graph.
                
                The only usecase for setting it to true is when actually changing
                `op` during the initialization of the optimizer (e.g. replacing it
                with MixedOp or SampleOp)
        """
        Graph._verify_update_function(update_func, private_edge_data)

        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or (graph.scope is not None and graph.scope in scope):
                logger.debug('Updating {}'.format(graph.name))
                for u, v, edge_data in graph.edges.data():
                    graph.edges[u, v].update(update_func(current_edge_data=edge_data))


    def clone(self):
        """
        Deep copy of the current graph.

        Returns:
            Graph: Deep copy of the graph.
        """
        return copy.deepcopy(self)


    def reset_weights(self, inplace: bool = False):
        """
        Resets the weights for the 'op' at all edges.

        Args:
            inplace (bool): Do the operation in place or
                return a modified copy.
        Returns:
            Graph: Returns the modified version of the graph.
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
