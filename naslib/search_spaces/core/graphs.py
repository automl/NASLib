from abc import ABCMeta, abstractmethod
from collections import Set, Mapping, MutableSet, MutableMapping, namedtuple
from functools import partial

Node = partial(namedtuple('Node', 'value label'), label=None)


class Graph(metaclass=ABCMeta):
    """Graph abstract base class.

    Edges are always labeled, with a default label of 1.

    The following invariants always apply:
    (n1, n2) in g.edges implies {n1, n2} <= g.nodes
    n not in g.nodes implies all(n not in edge for edge in g.edges)
    """

    class Nodes(Set):
        """ABC for the nodes."""

        def __init__(self, g):
            self.graph = g

    class Edges(Mapping):
        """ABC for the edges."""

        def __init__(self, g):
            self.graph = g

    def __init__(self):
        self.nodes = self.Nodes(self())
        self.edges = self.Edges(self())

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges

    def __ne__(self, other):
        return not (self == other)

    @abstractmethod
    def neighbors(self, node) -> Mapping:
        """Mapping from neighbors to label of corresponding edges.

        Do not try to modify the mapping resulting from this function:
        it may result in inconsistent behaviour.
        """

    @classmethod
    def copy_from(cls, g):
        """Returns a new graph with the same nodes and edges.

        An error is returned if a directed instance is called from an
        undirected one.
        """

        return cls(g.data)


class MutableGraph(Graph):

    class Nodes(MutableSet, Graph.Nodes):

        @abstractmethod
        def discard(self, node):
            """All edges incident to node should be discarded as well."""

    class Edges(MutableMapping, Graph.Edges):

        @abstractmethod
        def __setitem__(self, edge):
            """Nodes of the edge should be added to the graph's nodes."""

        # Convenience methods that add (default edge label: 1) and
        # discard edges mimicking a Set-like two-argument interface.

        def add(self, n1, n2):
            self[n1, n2] = 1

        def discard(self, n1, n2):
            try:
                del self[n1, n2]
            except KeyError:
                pass

    def neighbors(self, node):
        res = {}
        for (n1, n2), v in self.edges.items():
            if n1 == node:
                res[n2] = v
            elif n2 == node:
                res[n1] = v
        return res


class DirectedGraph(MutableGraph):
    """Graphs, represented using a dict-of-dicts representation.

    For example, a graph with:
     - nodes {0, 1, 2, 3, 4, 5}
     - edges {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 2), (4, 5)}
    and all edge labels equal to 1 is represented as
    {Node(0): {Node(1): 1, Node(2): 1},
     Node(1): {Node(2): 1, Node(3): 1},
     Node(2): {Node(3): 1},
     Node(3): {Node(2): 1},
     Node(4): {Node(5): 1},
     Node(5): {}}

    This representation can be accessed via the `data' attribute.

    For undirected dictionaries, edges are represented in both of nodes'
    adjacency mappings.
    """

    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data
        super(DirectedGraph, self).__init__()

        self.add_missing_nodes()

    def __call__(self):
        return self.data

    def get_nodes(self):
        return list(self.nodes)

    def get_edges(self):
        return list(self.edges)

    def get_node(self, node_id):
        node = list(filter(lambda x: x.value==node_id, self.get_nodes()))
        return node if len(node) == 0 else node[0]

    def get_node_ids(self):
        return [x.value for x in self.get_nodes()]

    def get_node_label(self, node_id):
        node = self.get_node(node_id)
        return None if node == [] else node.label

    def get_edge_label(self, edge):
        n1, n2 = edge
        if edge not in self.get_edges():
            raise IndexError('Edge {} not in Graph'.format(edge))
        _n1 = list(filter(lambda x: x.value==n1, self.data.keys()))[0]
        _n2 = list(filter(lambda x: x.value==n2, self.data[_n1]))[0]
        return self.data[_n1][_n2]

    def neighbors(self, node_id):
        return self.data[self.get_node(node_id)]

    def incoming(self, node_id):
        node = self.get_node(node_id)
        return {n: adj[node] for n, adj in self.data.items() if node in adj}

    def outgoing(self, node_id):
        node = self.get_node(node_id)
        return {n2: v for (n1, n2), v in self.edges.items() if n1 == node}

    def add_missing_nodes(self):
        existing_nodes = set(self.get_node_ids())
        union = existing_nodes.union(set(x[1] for x in self.get_edges()))
        for node_id in union.difference(existing_nodes):
            self.nodes.add(node_id, 'out')

    class Nodes(MutableGraph.Nodes):

        def __init__(self, data):
            self.data = data

        def add(self, node_id, label: str):
            self[node_id] = label

        def discard(self, node_id):
            node = self[node_id]
            if node in self.data:
                del self.data[node]
                for neighbordict in self.data.values():
                    if node in neighbordict:
                        del neighbordict[node]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def __contains__(self, node_id):
            return self[node_id] in self.data

        def __getitem__(self, node_id):
            node = list(filter(lambda x: x.value==node_id, list(self)))
            return node if len(node) == 0 else node[0]

        def __setitem__(self, node_id, label: str):
            node = self[node_id]
            if node == []:
                self.data.setdefault(Node(node_id, label=label), {})
            elif node in list(self):
                existing_node = list(filter(lambda x: x.value==node_id,
                                            list(self)))[0]
                if node.label == existing_node.label:
                    print('Node {} already exists'.format(node))
                else:
                    raise('Attempting to set Node {} label to '
                          '{}'.format(existing_node, node.label))

        @classmethod
        def _from_iterable(cls, it):
            # Needed by the Set abstract base class
            return set(it)
            # return cls({node: {} for node in it})

        def __le__(self, other):
            if isinstance(other, DirectedGraph):
                return self.data.keys() <= other.data.keys()
            return super().__le__(other)

        def __iand__(self, c):
            for node in self - c:
                self.discard(node)
            return self

    class Edges(MutableGraph.Edges):

        def __init__(self, data):
            self.data = data

        def __contains__(self, edge):
            n1, n2 = edge
            if isinstance(n1, tuple) and isinstance(n2, tuple):
                return n1 in self.data and n2 in self.data[n1]
            else:
                return edge in list(self)

        def __iter__(self):
            return ((n1.value, n2.value) for n1, ndict in self.data.items() for n2 in ndict)

        def __len__(self):
            return sum(map(len, self.data.values()))

        def __getitem__(self, edge):
            n1, n2 = edge
            _n1 = list(filter(lambda x: x.value==n1, self.data.keys()))[0]
            _n2 = list(filter(lambda x: x.value==n2, self.data[_n1]))[0]
            return self.data[_n1][_n2]

        def __setitem__(self, edge, value):
            n1, n2 = edge
            _n1 = list(filter(lambda x: x.value==n1, self.data.keys()))[0]
            _n2 = list(filter(lambda x: x.value==n2, self.data[_n1]))
            _n2 = Node(n2) if len(_n2) == 0 else _n2[0]
            self.data.setdefault(_n2, {})
            self.data.setdefault(_n1, {})[_n2] = value

        def __delitem__(self, edge):
            n1, n2 = edge
            _n1 = list(filter(lambda x: x.value==n1, self.data.keys()))[0]
            _n2 = list(filter(lambda x: x.value==n2, self.data[_n1]))[0]
            del self.data[n1][n2]


