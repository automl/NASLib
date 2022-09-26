import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from naslib.predictors import Predictor
from naslib.search_spaces.nasbench301.conversions import convert_naslib_to_genotype
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_op_indices

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OneShotPredictor(Predictor):
    def __init__(self, config, trainer, model_path=None):
        self.config = config
        self.model = trainer
        if trainer.optimizer.graph.get_type() == "nasbench301":
            self.converter = convert_naslib_to_genotype
        else:
            self.converter = convert_naslib_to_op_indices

        if model_path is None:
            # if no saved model is provided conduct the search from scratch.
            # NOTE: that this is an expensive step and it should be avoided when
            # using the oneshot model as performance predictor
            print("No saved model found! Starting search...")
            self.model.search()
        else:
            # TODO: change after refactoring checkpointer in NASLib
            print("Loading model from {}".format(model_path))
            self.model.optimizer.graph.load_state_dict(torch.load(model_path)["model"])
            print("Fineshed loading model")

    def __call__(self, archs):
        """
        Evaluate, i.e. do a forward pass for every image datapoint, the
        one-shot model for every architecture in archs.
            params:
                archs: torch.Tensor where each element is an architecture encoding

            return:
                torch.Tensor with the predictions for every arch in archs
        """
        prediction = []
        for arch in archs:
            # we have to iterate through all the architectures in the
            # mini-batch
            self.model.optimizer.set_alphas_from_path(arch)
            # NOTE: evaluation on the 25k validation data for now. provide a test
            # dataloader to evaluate on the test data
            val_acc = self.model.evaluate_oneshot(dataloader=None)
            prediction.append(val_acc)
        print("Predictions:")
        print(prediction)

        return prediction

    def fit(self, xtrain, ytrain, train_info=None, verbose=0):
        pass

    def query(self, xtest, info=None, eval_batch_size=None):
        _xtest = [self.converter(arch) for arch in xtest]
        return np.squeeze(np.array(self(_xtest)))
