import sys
print(sys.path)
from naslib.predictors.predictor import Predictor
from compute import count_parameters

class ZeroCostPredictor(Predictor):
    def __init__(self):
        self.method_type = 'count_params'

    def pre_process(self):
        pass

    def query(self, xtest, info=None):
        graph = xtest
        score = count_parameters(graph)
        # Higher the score, higher the ranking of the model
        return score
