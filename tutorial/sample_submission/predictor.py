from naslib.predictors.predictor import Predictor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class ZeroCostPredictor(Predictor):
    '''Your predictor MUST be named ZeroCostPredictor'''
    def __init__(self):
        self.method_type = 'count_params'

    def pre_process(self):
        pass

    def query(self, xtest, info=None):
        graph = xtest
        score = count_parameters(graph)
        # Higher the score, higher the ranking of the model
        return score