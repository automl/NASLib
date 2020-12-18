from naslib.predictors.predictor import Predictor


class EarlyStopping(Predictor):

    def __init__(self, dataset, metric):
        
        self.metric = metric
        self.dataset = dataset
    
    def query(self, xtest, info):
        return info
        
    def get_metric(self):
        return self.metric
    
    def requires_partial_training(self, xtest, fidelity):
        info = [arch.query(self.metric, self.dataset, epoch=fidelity) for arch in xtest]
        return info
