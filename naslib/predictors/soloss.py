from naslib.predictors.predictor import Predictor
import numpy as np

class SoLosspredictor(Predictor):
    
    def __init__(self, dataset, metric='train_loss', sum_option='SoTL'):
        self.metric = metric
        self.sum_option = sum_option
        self.name = 'SoLoss'
        self.dataset = dataset

    def query(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures, 
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        test_set_scores = []
        for test_arch, past_loss in zip(xtest, info):
            # assume we have the training loss for each preceding epoch: past_loss is a list
            
            if self.sum_option == 'SoTLE' or self.sum_option == 'SoVLE':
                score = past_loss[-1]
            elif 'EMASo' in self.sum_option:
                EMA_SoTL = []
                mu = 0.99
                for se in range(np.array(info).shape[1]):
                    if se <= 0:
                        ema = past_loss[se]
                    else:
                        ema = ema * (1 - mu) + mu * past_loss[se]
                    EMA_SoTL.append(ema)
                score = np.sum(EMA_SoTL)
            else:
                score = np.sum(past_loss)
            test_set_scores.append(-score)
            
        return np.array(test_set_scores)
    
    def requires_partial_training(self, xtest, fidelity):
        info = [arch.query(self.metric, self.dataset, epoch=200)[:fidelity] for arch in xtest]
        return info