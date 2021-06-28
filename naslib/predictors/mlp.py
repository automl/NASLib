import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from naslib.utils.utils import AverageMeterGroup
from naslib.predictors.utils.encodings import encode
from naslib.predictors import Predictor

# NOTE: faster on CPU
device = torch.device('cpu')
print('device:', device)

def accuracy_mse(prediction, target, scale=100.):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class FeedforwardNet(nn.Module):
    def __init__(self, input_dims: int = 5, num_layers: int = 3, layer_width:
                 list = [10, 10, 10], output_dims: int = 1, activation='relu'):
        super(FeedforwardNet, self).__init__()
        assert len(layer_width) == num_layers, "number of widths should be \
        equal to the number of layers"

        self.activation = eval('F.'+activation)

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList([nn.Linear(all_units[i], all_units[i+1]) for i in
                                     range(num_layers)])

        self.out = nn.Linear(all_units[-1], 1)

        # make the init similar to the tf.keras version
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out(x)

    def basis_funcs(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class MLPPredictor(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', hpo_wrapper=False):
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {'num_layers': 20, 
                                    'layer_width': 20, 
                                    'batch_size': 32,
                                    'lr': 0.001,
                                    'regularization': 0.2}
        self.hyperparams = None

    def get_model(self, **kwargs):
        predictor = FeedforwardNet(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain, train_info=None,
            epochs=500, loss='mae', verbose=0):
        
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        num_layers = self.hyperparams['num_layers']
        layer_width = self.hyperparams['layer_width']
        batch_size = self.hyperparams['batch_size']
        lr = self.hyperparams['lr']
        regularization = self.hyperparams['regularization']

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

        self.model = self.get_model(input_dims=_xtrain.shape[1],
                                    num_layers=num_layers,
                                    layer_width=num_layers*[layer_width])
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        if loss == 'mse':
            criterion = nn.MSELoss().to(device)
        elif loss == 'mae':
            criterion = nn.L1Loss().to(device)

        self.model.train()

        for e in range(epochs):
            meters = AverageMeterGroup()
            for b, batch in enumerate(data_loader):
                optimizer.zero_grad()
                input = batch[0].to(device)
                target = batch[1].to(device)
                prediction = self.model(input).view(-1)

                loss_fn = criterion(prediction, target)
                # add L1 regularization
                params = torch.cat([x[1].view(-1) for x in
                                    self.model.named_parameters() if x[0] == 'out.weight'])
                loss_fn += regularization * torch.norm(params, 1)
                loss_fn.backward()
                optimizer.step()

                mse = accuracy_mse(prediction, target)
                meters.update({"loss": loss_fn.item(), "mse": mse.item()}, n=target.size(0))

            if verbose and e%100 == 0:
                print('Epoch {}, {}, {}'.format(e, meters['loss'], meters['mse']))

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

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch[0].to(device)).view(-1)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred)

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                'num_layers': int(np.random.choice(range(5,25))),
                'layer_width': int(np.random.choice(range(5,25))),
                'batch_size': 32,
                'lr': np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                'regularization': 0.2}

        self.hyperparams = params
        return params