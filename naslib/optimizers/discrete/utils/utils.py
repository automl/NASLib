import numpy as np


"""
These two function discretize the graph.
"""
def add_sampled_op_index(edge):
    """
    Function to sample an op for each edge.
    """
    op_index = np.random.randint(len(edge.data.op))
    edge.data.set('op_index', op_index, shared=True)


def update_ops(edge):
    """
    Function to replace the primitive ops at the edges
    with the sampled one
    """
    if isinstance(edge.data.op, list):
        primitives = edge.data.op
    else:
        primitives = edge.data.primitives
    edge.data.set('op', primitives[edge.data.op_index])
    edge.data.set('primitives', primitives)     # store for later use


def sample_random_architecture(search_space, scope):
    architecture = search_space.clone()

    # We are discreticing here so
    architecture.prepare_discretization()

    # 1. add the index first (this is shared!)
    architecture.update_edges(
        add_sampled_op_index,
        scope=scope,
        private_edge_data=False
    )

    # 2. replace primitives with respective sampled op
    architecture.update_edges(
        update_ops, 
        scope=scope,
        private_edge_data=True
    )
    return architecture


def get_op_indices(arch):
    # used for debugging
    cells = arch._get_child_graphs(single_instances=True)
    op_indices = []
    for cell in cells:
        edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]
    for edge in edges:
        op_indices.append(cell.edges[edge].op_index)
        
    return op_indices


def mutate(parent_arch):
    child = parent_arch.clone()
        
    # sample which cell/motif we want to mutate
    cells = child._get_child_graphs(single_instances=True)
    cell = np.random.choice(cells) if len(cells) > 1 else cells[0]
        
    edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]

    # sample if op or edge change
    # note: the "change edge" is broken. But for nb201, we only need "change op"
    if True:
        # change op
        random_edge = edges[np.random.choice(len(edges))]
        data = cell.edges[random_edge]
        available = [o for o in range(len(data.primitives)) if o != data.op_index]
        op_index = np.random.choice(available)
        data.set('op_index', op_index, shared=True)
    else:
        # change edge by setting it to zero
        random_edge = edges[np.random.choice(len(edges))]
        cell.edges[random_edge].set('op_index', 1, shared=True)     # this is search space dependent

        random_edge = edges[np.random.choice(len(edges))]
        data = cell.edges[random_edge]
        op_index = np.random.randint(len(data.primitives))
        cell.edges[random_edge].set('op_index', op_index, shared=True)
        
    child.update_edges(update_ops, child.OPTIMIZER_SCOPE, private_edge_data=True)
    return child

