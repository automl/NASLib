from naslib.predictors.predictor import Predictor


class EarlyStopping(Predictor):

    def __init__(self, metric):
        
        self.metric = metric
    
    def get_fidelity(self):
        return self.fidelity
    
    def get_metric(self):
        return self.metric
    
    def query(self, xtest, info):
        return info
        
    def get_metric(self):
        return self.metric

    def get_type(self):
        return 'partial_training'
    
    # to be changed to another method
    def requires_partial_training(self):
        return True
    
    # to be deleted
    def get_fidelity(self):
        return self.fidelity
