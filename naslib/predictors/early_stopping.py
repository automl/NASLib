from naslib.predictors.predictor import Predictor


class EarlyStopping(Predictor):

    def __init__(self, fidelity, metric):
        
        # fidelity is the number of epochs to train for
        self.fidelity = fidelity
        self.metric = metric
    
    def get_fidelity(self):
        return self.fidelity
    
    def get_metric(self):
        return self.metric
    
    def query(self, xtest, info):
        return info
    
    def requires_partial_training(self):
        return True
    
    def get_fidelity(self):
        return self.fidelity
    
    def get_metric(self):
        return self.metric
