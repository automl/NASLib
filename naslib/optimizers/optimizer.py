from naslib.search_spaces.core.operations import CategoricalOp


class Optimizer:
    def __init__(self):
        self.architectural_weights = []

    def replace_function(self, graph_obj):
        if 'op_choices' in graph_obj:
            graph_obj['op'] = CategoricalOp(primitives=graph_obj['op_choices'], **graph_obj['op_kwargs'])
        return graph_obj
