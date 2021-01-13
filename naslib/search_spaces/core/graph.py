import networkx as nx
import copy
import logging
import torch
import random
import inspect

from networkx.algorithms.dag import lexicographical_topological_sort

from naslib.utils.utils import iter_flatten, AttrDict
from naslib.utils.logging import log_formats, log_first_n
from .primitives import Identity, AbstractPrimitive
from .query_metrics import Metric

logger = logging.getLogger(__name__)


class Graph(torch.nn.Module, nx.DiGraph):
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
    it has been defined already use the function `update_edges()`
    or `update_nodes()`.

    **Use as pytorch module**
    If you want to learn the weights of the operations or any
    other parameters of the graph you have to parse it first.
    >>> graph = getFancySearchSpace()
    >>> graph.parse()
    >>> logits = graph(data)
    >>> optimizer.min(loss(logits, target))

    To update the pytorch module representation (e.g. after removing or adding
    some new edges), you have to unparse. Beware that this is not fast, so it should
    not be done on each batch or epoch, rather once after discretizising. If you
    want to change the representation of the graph use rather some shared operation
    indexing at the edges.
    >>> graph.update(remove_random_edges)
    >>> graph.unparse()
    >>> graph.parse()
    >>> logits = graph(data)

    **Interface to tabular benchmarks**
    TODO
    """

    """
    Usually the optimizer does not operate on the whole graph, e.g. preprocessing
    and post-processing are excluded. Scope can be used to define that or to
    differentate instances of the "same" graph.
    """
    OPTIMIZER_SCOPE = "all"

    """
    Whether the search space has an interface to one of the tabular benchmarks which
    can then be used to query architecture performances.

    If this is set to true then `query()` should be implemented.
    """
    QUERYABLE = False


    def __init__(self, name: str = None, scope: str = None):
        """
        Initialise a graph. The edges are automatically filled with an EdgeData object
        which defines the default operation as Identity. The default combination operation
        is set as sum.

        Note:
            When inheriting form `Graph` note that `__init__()` cannot take any parameters.
            This is due to the way how networkx is implemented, i.e. graphs are reconstructed
            internally and no parameters for init are considered.

            Our recommended solution is to create static attributes before initialization and
            then load them dynamically in `__init__()`.

            >>> def __init__(self):
            >>>     num_classes = self.NUM_CLASSES
            >>> MyGraph.NUM_CLASSES = 42
            >>> my_graph_42_classes = MyGraph()

        """
        #super().__init__()
        nx.DiGraph.__init__(self)
        torch.nn.Module.__init__(self)

        # Make DiGraph a member and not inherit. This is because when inheriting from
        # `Graph` note that `__init__()` cannot take any parameters. This is due to 
        # the way how networkx is implemented, i.e. graphs are reconstructed internally
        # and no parameters for init are considered.
        # Therefore __getattr__ and __iter__ forward the DiGraph methods for straight-forward 
        # usage as if we would inherit.

        #self._nxgraph = nx.DiGraph()

        # Replace the default dicts at the edges with `EdgeData` objects
        # `EdgeData` can be easily customized and allow shared parameters
        # across different Graph instances.

        #self._nxgraph.edge_attr_dict_factory = lambda: EdgeData()
        self.edge_attr_dict_factory = lambda: EdgeData()

        # Replace the default dicts at the nodes to include `input` from the beginning.
        # `input` is required for storing the results of incoming edges.

        #self._nxgraph.node_attr_dict_factory = lambda: dict({'input': {}, 'comb_op': sum})
        self.node_attr_dict_factory = lambda: dict({'input': {}, 'comb_op': sum})

        # remember to add all members also in `unparse()`
        self.name = name
        self.scope = scope
        self.input_node_idxs = None
        self.is_parsed = False
        self._id = random.random()    # pytorch expects unique modules in `add_module()`


    def __eq__(self, other):
        return self.name == other.name and self.scope == other.scope


    def __hash__(self):
        """
        As it is very complicated to compare graphs (i.e. check all edge
        attributes, do the have shared attributes, ...) use just the name
        for comparison.

        This is used when determining whether two instances are copies.
        """
        h = 0
        h += hash(self.name)
        h += hash(self.scope) if self.scope else 0
        h += hash(self._id)
        return h


    def __repr__(self):
        return "Graph {}-{:.07f}, scope {}, {} nodes".format(self.name, self._id, self.scope, self.number_of_nodes())


    def modules_str(self):
        """
        Once the graph has been parsed, prints the modules as they appear in pytorch.
        """
        if self.is_parsed:
            result = ""
            for g in self._get_child_graphs(single_instances=True) + [self]:
                result += "Graph {}:\n {}\n==========\n".format(g.name, torch.nn.Module.__repr__(g))
            return result
        else:
            return self.__repr__()


    def set_scope(self, scope: str, recursively=True):
        """
        Sets the scope of this instance of the graph.

        The function should be used in a builder-like pattern
        `'subgraph'=Graph().set_scope("scope")`.

        Args:
            scope (str): the scope
            recursively (bool): Also set the scope for all child graphs.
                default True

        Returns:
            Graph: self with the setted scope.
        """
        self.scope = scope
        if recursively:
            for g in self._get_child_graphs(single_instances=False):
                g.scope = scope
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
        nx.DiGraph.add_node(self, node_index, **attr)


    def copy(self):
        """
        Copy as defined in networkx, i.e. a shallow copy.

        Just handling recursively nested graphs seperately.
        """

        def copy_dict(d):
            copied_dict = d.copy()
            for k, v in d.items():
                if isinstance(v, Graph):
                    copied_dict[k] = v.copy()
                elif isinstance(v, list):
                    copied_dict[k] = [i.copy() if isinstance(i, Graph) else i for i in v]
                elif isinstance(v, torch.nn.Module) or isinstance(v, AbstractPrimitive):
                    copied_dict[k] = copy.deepcopy(v)
            return copied_dict

        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, copy_dict(d)) for n, d in self._node.items())
        G.add_edges_from((u, v, datadict.copy())
                         for u, nbrs in self._adj.items()
                         for v, datadict in nbrs.items())
        G.scope = self.scope
        G.name = self.name
        return G


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
            for node_idx in lexicographical_topological_sort(self):
                if self.in_degree(node_idx) == 0:
                    self.nodes[node_idx]['input'] = {0: x[next(input_node_iterator)]}


    def forward(self, x, *args):
        """
        Forward some data through the graph. This is done recursively
        in case there are graphs defined on nodes or as 'op' on edges.

        Args:
            x (Tensor or dict): The input. If the graph sits on a node the
                input can be a dict with {source_idx: Tensor} to be routed
                to the defined input nodes. If the graph sits on an edge,
                x is the feature tensor.
            args: This is only required to handle cases where the graph sits
                on an edge and receives an EdgeData object which will be ignored
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
            node['input'] = {}  # clear the input as we have processed it

            if len(list(self.neighbors(node_idx))) == 0 and node_idx < list(lexicographical_topological_sort(self))[-1]:
                # We have more than one output node. This is e.g. the case for
                # auxillary losses. Attach them to the graph, handling must done
                # by the user.
                logger.debug("Graph {} has more then one output node. Storing output of non-maximum index node {} at graph dict".format(
                    self, node_idx
                ))
                self.graph['out_from_{}'.format(node_idx)] = x
            else:
                # outgoing edges: process all outgoing edges
                for neigbor_idx in self.neighbors(node_idx):
                    edge_data = self.get_edge_data(node_idx, neigbor_idx)
                    # inject edge data only for AbstractPrimitive, not Graphs
                    if isinstance(edge_data.op, Graph):
                        edge_output = edge_data.op.forward(x)
                    elif isinstance(edge_data.op, AbstractPrimitive):
                        logger.debug("Processing op {} at edge {}-{}".format(edge_data.op, node_idx, neigbor_idx))
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
                elif edge_data.op.get_embedded_ops():
                    for primitive in edge_data.op.get_embedded_ops():
                        if isinstance(primitive, Graph):
                            primitive.parse()
                self.add_module("{}-edge({},{})".format(self.name, node_idx, neigbor_idx), edge_data.op)
        self.is_parsed = True


    def unparse(self):
        """
        Undo the pytorch parsing by reconstructing the graph uusing the
        networkx data structures.

        This is done recursively also for child graphs.

        Returns:
            Graph: An unparsed shallow copy of the graph.
        """
        g = self.__class__()
        g.clear()

        graph_nodes = self.nodes
        graph_edges = self.edges

        # unparse possible child graphs
        # be careful with copying/deepcopying here cause of shared edge data
        for n, data in graph_nodes.data():
            if 'subgraph' in data:
                data['subgraph'] = data['subgraph'].unparse()
        for u, v, data in graph_edges.data():
            if isinstance(data.op, Graph):
                data.set('op', data.op.unparse())

        # create the new graph
        # Remember to add all members here to update. I know it is ugly but don't know better
        g.add_nodes_from(graph_nodes.data())
        g.add_edges_from(graph_edges.data())
        g.graph.update(self.graph)
        g.name = self.name
        g.input_node_idxs = self.input_node_idxs
        g.scope = self.scope
        g.is_parsed = False
        g._id = self._id
        g.OPTIMIZER_SCOPE = self.OPTIMIZER_SCOPE
        g.QUERYABLE = self.QUERYABLE

        return g


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
                    if isinstance(embedded_ops, Graph):
                        graphs.append(embedded_ops)
                        graphs.append(embedded_ops._get_child_graphs())
                    elif isinstance(embedded_ops, list):
                        for child_op in edge_data.op.get_embedded_ops():
                            if isinstance(child_op, Graph):
                                graphs.append(child_op)
                                graphs.append(child_op._get_child_graphs())
                    else:
                        logger.debug("Got embedded op, but is neither a graph nor a list: {}".format(embedded_ops))
            elif inspect.isclass(edge_data.op):
                assert not issubclass(edge_data.op, Graph), "Found non-initialized graph. Abort."
                pass    # we look at an uncomiled op
            else:
                raise ValueError("Unknown format of op: {}".format(edge_data.op))

        graphs = [g for g in iter_flatten(graphs)]

        if single_instances:
            single = []
            for g in graphs:
                if g.name not in [sg.name for sg in single]:
                    single.append(g)
            return sorted(single, key=lambda g: g.name)
        else:
            return sorted(graphs, key=lambda g: g.name)


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
        assert scope is not None
        result = []
        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or graph.scope == scope or (isinstance(scope, list) and graph.scope in scope):
                for u, v, edge_data in graph.edges.data():
                    if edge_data.has(key):
                        result.append(edge_data[key])
        return result


    def set_at_edges(self, key, value, shared=False):
        """
        Sets the attribute for all edges in this and any child graph
        """
        for graph in self._get_child_graphs(single_instances=shared) + [self]:
            logger.debug('Updating edges of graph {}'.format(graph.name))
            for _, _, edge_data in graph.edges.data():
                if not edge_data.is_final():
                    edge_data.set(key, value, shared)


    def compile(self):
        """
        Instanciates the ops at the edges using the arguments specified at the edges
        """
        for graph in self._get_child_graphs(single_instances=False) + [self]:
            logger.debug('Compiling graph {}'.format(graph.name))
            for u, v, edge_data in graph.edges.data():
                if not edge_data.is_final():
                    attr = edge_data.to_dict()
                    op = attr.pop('op')

                    if isinstance(op, list):
                        compiled_ops = []
                        for i, o in enumerate(op):
                            if inspect.isclass(o):
                                # get the relevant parameter if there are more.
                                a = {k:v[i] if isinstance(v, list) else v for k, v in attr.items()}
                                compiled_ops.append(o(**a))
                            else:
                                logger.debug("op {} already compiled. Skipping".format(o))
                        edge_data.set('op', compiled_ops)
                    elif isinstance(op, AbstractPrimitive):
                        logger.debug("op {} already compiled. Skipping".format(op))
                    elif inspect.isclass(op) and issubclass(op, AbstractPrimitive):
                        # Init the class
                        edge_data.set('op', op(**attr))
                    elif isinstance(op, Graph):
                        pass  # This is already covered by _get_child_graphs
                    else:
                        raise ValueError("Unkown format of op: {}".format(op))


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
        test.set('op', [True])

        try:
            result = test.clone()
            update_func(current_edge_data=result)
        except:
            log_first_n(logging.WARN, "Update function could not be veryfied. Be cautious with the "
                "setting of `private_edge_data` in `update_edges()`", n=5)
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

        Note that edges marked as 'final' will not be updated here.

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
        assert scope is not None
        for graph in self._get_child_graphs(single_instances=not private_edge_data) + [self]:
            if scope == 'all' or scope == graph.scope or (isinstance(scope, list) and graph.scope in scope):
                logger.debug('Updating edges of graph {}'.format(graph.name))
                for u, v, edge_data in graph.edges.data():
                    if not edge_data.is_final():
                        edge = AttrDict(head=u, tail=v, data=edge_data)
                        update_func(edge=edge)
        self._delete_flagged_edges()


    def update_nodes(self, update_func: callable, scope="all", single_instances: bool = True):
        """
        Update the nodes of the graph and its incoming and outgoing edges by iterating over the
        graph and applying `update_func` to each of it. This is the
        preferred way to change the search space once it has been defined.

        Note that edges marked as 'final' will not be updated here.

        Args:
            update_func (callable): Function that accepts three incoming parameters named
                `node, in_edges, out_edges`.
                    - `node` is a tuple (int, dict) containing the
                      index and the attributes of the current node.
                    - `in_edges` is a list of tuples with the index of
                      the tail of the edge and its EdgeData.
                    - `out_edges is a list of tuples with the index of
                      the head of the edge and its EdgeData.
            scope (str or list(str)): Can be "all" or list of scopes to be updated. Only graphs
                and child graphs with the specified scope are considered
            single_instance (bool): If set to false, this means update_func will be
                applied to nodes of all copies of a graphs. THIS IS NOT RECOMMENDED FOR SHARED
                ATTRIBUTES, i.e. when manipulating the shared data of incoming or outgoing edges.
                Shared attributes should be set only once, we take care it is syncronized across
                all copies of this graph.

                The only usecase for setting it to true is when actually changing
                `op` during the initialization of the optimizer (e.g. replacing it
                with MixedOp or SampleOp)
        """
        assert scope is not None
        for graph in self._get_child_graphs(single_instances) + [self]:
            if scope == 'all' or graph.scope == scope or (isinstance(scope, list) and graph.scope in scope):
                logger.debug('Updating nodes of graph {}'.format(graph.name))
                for node_idx in lexicographical_topological_sort(graph):
                    node = (node_idx, graph.nodes[node_idx])
                    in_edges = list(graph.in_edges(node_idx, data=True))                        # (v, u, data)
                    in_edges = [(v, data) for v, u, data in in_edges if not data.is_final()]    # u is same for all
                    out_edges = list(graph.out_edges(node_idx, data=True))                      # (v, u, data)
                    out_edges = [(u, data) for v, u, data in out_edges if not data.is_final()]  # v is same for all
                    update_func(node=node, in_edges=in_edges, out_edges=out_edges)
        self._delete_flagged_edges()


    def _delete_flagged_edges(self):
        """
        Delete edges which associated EdgeData is flagged as deleted.
        """
        for graph in self._get_child_graphs(single_instances=False) + [self]:    # we operate on shallow copies
            to_remove = []
            for u, v, edge_data in graph.edges.data():
                if edge_data.is_deleted():
                    to_remove.append((u, v))
            if to_remove:
                # logger.info("Removing edges {} from graph {}".format(to_remove, graph))
                graph.remove_edges_from(to_remove)


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

        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        if inplace:
            graph = self
        else:
            graph = self.clone()

        graph.apply(weight_reset)

        return graph


    def prepare_discretization(self):
        """
        In some cases the search space is manipulated before the final
        discretization is happening, e.g. DARTS. In such chases this should
        be defined in the search space, so all optimizers can call it.
        """
        pass


    def prepare_evaluation(self):
        """
        In some cases the evaluation architecture does not match the searched
        one. An example is where the makro_model is extended to increase the
        parameters. This is done here.
        """
        pass

    
    def sample_random_architecture(self, dataset_api):
        raise NotImplementedError()


    def mutate(self):
        raise NotImplementedError()


    def query(self, metric: Metric, dataset: str, path: str) -> float:
        """
        Can be used to query the performance of the architecture using
        a tabular benchmark.

        The interface must be provided by the search space.

        Args:
            metric (str): the name of the metric to query.
            dataset (str): the name of the dataset to query.
            path (str): path where the tabular data is located

        Returns:
            float: The performance number
        """
        if self.QUERYABLE:
            raise NotImplementedError("QUERYABLE set to True therefore query_performance must be implemented")
        else:
            raise NotImplementedError("This function should not be used if QUERYABLE is False")


    def get_dense_edges(self):
        """
        Returns the edge indices (i, j) that would make a fully connected
        DAG without circles such that i < j and i != j. Assumes nodes are
        already created.

        Returns:
            list: list of edge indices.
        """
        edges = []
        nodes = sorted(list(self.nodes()))
        for i in nodes:
            for j in nodes:
                if i != j and j > i:
                    edges.append((i, j))
        return edges


    def add_edges_densly(self):
        """
        Adds edges to get a fully connected DAG without cycles
        """
        self.add_edges_from(self.get_dense_edges())


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

        # set internal attributes
        self._shared['_deleted'] = False
        self._private['_final'] = False

        # set defaults and potential input
        self.set('op', Identity(), shared=False)
        for k, v in data.items():
            self.set(k, v, shared=False)


    def has(self, key: str):
        """
        Checks whether `key` exists.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if key exists, False otherwise.

        """
        assert not key.startswith("_"), "Access to private keys not allowed!"
        if key in self._private.keys() or key in self._shared.keys():
            return True
        else:
            return False


    def __getitem__(self, key: str):
        assert not key.startswith("_"), "Access to private keys not allowed!"
        return self.__getattr__(key)


    def get(self, key: str, default):
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default


    def __getattr__(self, key: str):
        if key.startswith("__"):       # Required for deepcopy, not sure why
            raise AttributeError(key)  # 
        assert not key.startswith("_"), "Access to private keys not allowed!"
        if key in self._private:
            return self._private[key]
        elif key in self._shared:
            return self._shared[key]
        else:
            raise AttributeError("Cannot find field '{}' in the given EdgeData!".format(key))


    def __setattr__(self, name: str, val):
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


    def remove(self, key: str):
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

        # we need to handle copy of graphs seperately
        for k, v in self._private.items():
            if isinstance(v, Graph):
                new_self._private[k] = v.copy()
            elif isinstance(v, list):
                new_self._private[k] = [i.copy() if isinstance(i, Graph) else i for i in v]

        return new_self


    def set(self, key: str, value, shared=False):
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
        assert not key.startswith("_"), "Access to private keys not allowed!"
        assert not self.is_final(), "Trying to change finalized edge!"
        if shared:
            if key in self._private:
                raise ValueError("Key {} alredy defined as non-shared")
            else:
                self._shared[key] = value
        else:
            if key in self._shared:
                raise ValueError("Key {} alredy defined as shared".format(key))
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


    def delete(self):
        """
        Flag to delete the edge where this instance is attached to.
        """
        self._shared['_deleted'] = True


    def is_deleted(self):
        """
        Returns true if the edge is flagged to be deleted
        """
        return self._shared['_deleted']


    def finalize(self):
        """
        Sets this edge as final. This means it cannot be changed
        anymore and will also not appear in the update functions
        of the graph.
        """
        self._private['_final'] = True
        return self


    def is_final(self):
        """
        Returns:
            bool: True if the edge was finalized, False else
        """
        return self._private['_final']


    def to_dict(self, subset='all'):
        if subset == 'shared':
            return {k: v for k, v in self._shared.items() if not k.startswith('_')}
        elif subset == 'private':
            return {k: v for k, v in self._private.items() if not k.startswith('_')}
        elif subset == 'all':
            d = self.to_dict('private')
            d.update(self.to_dict('shared'))
            return d
        else:
            raise ValueError("Unknown subset {}".format(subset))

