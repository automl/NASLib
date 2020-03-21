from abc import ABCMeta, abstractmethod
from collections import Set, Mapping, MutableSet, MutableMapping


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
    {0: {1: 1, 2: 1},
     1: {2: 1, 3: 1},
     2: {3: 1},
     3: {2: 1},
     4: {5: 1},
     5: {}}

    This representation can be accessed via the `data' attribute.

    For undirected dictionaries, edges are represented in both of nodes'
    adjacency mappings.
    """

    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data
        super(DirectedGraph, self).__init__()

    def __call__(self):
        return self.data

    def neighbors(self, node):
        return self.data[node]

    def incoming(self, node):
        return {n: adj[node] for n, adj in self.data.items() if node in adj}

    def outgoing(self, node):
        return {n2: v for (n1, n2), v in self.edges.items() if n1 == node}

    class Nodes(MutableGraph.Nodes):

        def __init__(self, data):
            self.data = data

        def add(self, node):
            self.data.setdefault(node, {})

        def discard(self, node):
            data = self.data
            if node in data:
                del data[node]
                for neighbordict in data.values():
                    if node in neighbordict:
                        del neighbordict[node]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def __contains__(self, node):
                return node in self.data

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
            for node in  self - c:
                self.discard(node)
            return self

    class Edges(MutableGraph.Edges):

        def __init__(self, data):
            self.data = data

        def __contains__(self, edge):
            n1, n2 = edge
            return n1 in self.data and n2 in self.data[n1]

        def __getitem__(self, edge):
            n1, n2 = edge
            return self.data[n1][n2]

        def __len__(self):
            return sum(map(len, self.data.values()))

        def __setitem__(self, edge, value):
            n1, n2 = edge
            data = self.data
            data.setdefault(n2, {})
            data.setdefault(n1, {})[n2] = value

        def __delitem__(self, edge):
            n1, n2 = edge
            del self.data[n1][n2]

        def __iter__(self):
            return ((n1, n2) for n1, ndict in self.data.items() for n2 in ndict)


