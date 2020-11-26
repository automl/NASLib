class Predictor:
    
    def __init__(self):
        self.encoding = None
        self.metric = None
        self.fidelity = None
        
    def pre_process(self):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        pass
    
    def fit(self, xtrain, ytrain):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass
    
    def query(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures, 
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        pass
    
