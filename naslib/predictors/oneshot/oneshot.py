import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from naslib.predictors.utils.encodings import encode
from naslib.predictors import Predictor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


class OneShotPredictor(Predictor):

    def __init__(self, config, trainer, encoding_type='adjacency_one_hot',
                 model_path=None):
        self.config = config
        self.model = trainer
        self.encoding_type = encoding_type
        if model_path is None:
            self.model.search()
        else:
            pass





    def __call__(self, archs):
        prediction = []
        for arch in archs:
            # we have to iterate through all the architectures in the
            # mini-batch
            self.model.optimizer.set_alphas_from_path(arch)
            # NOTE: evaluation on the 25k validation data for now. provide a test
            # dataloader to evaluate on the test data
            self.model.evaluate_oneshot(dataloader=None)
            prediction.append(self.model.errors_dict.valid_acc)

        return torch.Tensor(prediction)


    def fit(self, xtrain, ytrain, train_info=None,
            verbose=0):

        #NOTE: the train data here is not used at all to train the predictor
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        _xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        _ytrain = np.array(ytrain)

        X_tensor = torch.FloatTensor(_xtrain).to(device)
        y_tensor = torch.FloatTensor(_ytrain).to(device)

        train_data = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True, drop_last=False,
                                 pin_memory=False)

        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error


    def query(self, xtest, info=None, eval_batch_size=None):
        xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                          ss_type=self.ss_type) for arch in xtest])
        X_tensor = torch.FloatTensor(xtest).to(device)
        test_data = TensorDataset(X_tensor)

        eval_batch_size = len(xtest) if eval_batch_size is None else eval_batch_size
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size,
                                      pin_memory=False)

        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self(batch[0]).view(-1)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred)


