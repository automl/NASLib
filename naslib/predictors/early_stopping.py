from naslib.predictors.predictor import Predictor


class EarlyStopping(Predictor):

    def __init__(self, dataset, fidelity, metric):
        
        # fidelity is the number of epochs to train for
        self.fidelity = fidelity
        self.metric = metric
        self.dataset = dataset
    
    def query(self, xtest, info):
        return info
    
    def requires_partial_training(self, xtest):
        info = [arch.query(self.metric, self.dataset, epoch=self.fidelity) for arch in xtest]
        return info
