import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from naslib.utils.utils import AverageMeterGroup
from naslib.predictors.utils.encodings import encode
from naslib.predictors import Predictor

from naslib.defaults.trainer import Trainer
from naslib.optimizers import OneShotOptimizer, RandomNASOptimizer
from naslib.search_spaces import DartsSearchSpace, NasBench201SearchSpace
from naslib.utils import utils, setup_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


class OneShotPredictor(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_model(self, model_ckpt, nas_optimizer='oneshot', **kwargs):
        config = utils.get_config_from_args()
        utils.set_seed(config.seed)
        utils.log_args(config)

        logger = setup_logger(config.save + "/log.log")
        logger.setLevel(logging.INFO)

        is self.ss_type == 'nasbench201':
            search_space = NasBench201SearchSpace()
        elif self.ss_type == 'darts':
            search_space = DartsSearchSpace()

        if nas_optimizer == 'oneshot':
            optimizer = OneShotNASOptimizer(config)
        elif nas_optimizer == 'rsws':
            optimizer = RandomNASOptimizer(config)
        optimizer.adapt_search_space(search_space)

        trainer = Trainer(optimizer, config, lightweight_output=True)
        if model_ckpt is not None:
            print('Loading one-shot model from {}'.format(model_ckpt))
            #TODO add the loading 
        else:
            print('No saved model found! Starting to train the one-shot model.')
            trainer.search()
            print('Finished training one-shot predictor.')

        return trainer


    def fit(self, xtrain, ytrain,
            model_path=None,
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

        self.model = self.get_model(model_path, nas_optimizer='oneshot')

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
                # should be torch.Tensor of shape [12]
                prediction = self.model(batch[0].to(device)).view(-1)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred)


    def set_arch_and_evaluate(self, archs):
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
