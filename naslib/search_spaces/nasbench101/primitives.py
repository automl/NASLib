from naslib.search_spaces.core.primitives import AbstractPrimitive

class ModelWrapper(AbstractPrimitive):
    def __init__(self, model):
        super().__init__(locals())
        self.model = model

    def get_embedded_ops(self):
        return None

    def forward(self, x, edge_data):
        return self.model(x)

    forward_beforeGP = forward

