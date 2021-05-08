# Author: Xingchen Wan & Binxin Ru @ University of Oxford
# Ru, B., Wan, X., et al., 2021. "Interpretable Neural Architecture Search via Bayesian Optimisation using Weisfeiler-Lehman Kernels". In ICLR 2021.


from copy import deepcopy

from grakel.utils import graph_from_networkx

from naslib.predictors.gp import BaseGPModel
from naslib.predictors.gp.gpwl_utils.convert import *
from naslib.predictors.gp.gpwl_utils.vertex_histogram import CustomVertexHistogram
from naslib.predictors.gp.gpwl_utils.wl_kernel import WeisfeilerLehman


def _normalize(y):
    y_mean = torch.mean(y)
    y_std = torch.std(y)
    y = (y - y_mean) / y_std
    return y, y_mean, y_std


def _transform(y):
    """By default naslib returns target in terms of accuracy in percentages. We transform this into
    log (error) in decimal"""
    return np.log(1. - np.array(y) / 100.)


def _untransform(y):
    """Inverse operation of _transform(y)"""
    return 100. * (1. - np.exp(y))


def unnormalize_y(y, y_mean, y_std, scale_std=False):
    """Similar to the undoing of the pre-processing step above, but on the output predictions"""
    if not scale_std:
        y = y * y_std + y_mean
    else:
        y *= y_std
    return y


def _compute_pd_inverse(K, jitter=1e-5):
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion.
    Return both the inverse matrix and the log determinant."""
    n = K.shape[0]
    # assert isinstance(jitter, float) or jitter.ndim == 0, 'only homoscedastic noise variance is allowed here!'
    is_successful = False
    fail_count = 0
    max_fail = 3
    while fail_count < max_fail and not is_successful:
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10 ** fail_count
            K_ = K + jitter_diag
            Kc = torch.cholesky(K_)
            is_successful = True
        except RuntimeError:
            fail_count += 1
    if not is_successful:
        print(K)
        raise RuntimeError("Gram matrix not positive definite despite of jitter")
    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.float(), logDetK.float()


def _compute_log_marginal_likelihood(K_i, logDetK, y, normalize=True, ):
    """Compute the zero mean Gaussian process log marginal likelihood given the inverse of Gram matrix K(x2,x2), its
    log determinant, and the training label vector y.
    Option:

    normalize: normalize the log marginal likelihood by the length of the label vector, as per the gpytorch
    routine.
    """
    # print(K_i.device, logDetK.device, y.device)
    lml = -0.5 * y.t() @ K_i @ y + 0.5 * logDetK - y.shape[0] / 2. * torch.log(2 * torch.tensor(np.pi, ))
    return lml / y.shape[0] if normalize else lml


class GraphGP:
    def __init__(self, xtrain, ytrain, gkernel,
                 space='nasbench101',
                 h='auto',
                 noise_var=1e-3,
                 num_steps=200,
                 max_noise_var=1e-1,
                 max_h=3,
                 optimize_noise_var=True,
                 node_label='op_name'
                 ):
        self.likelihood = noise_var
        self.space = space
        self.h = h

        if gkernel == 'wl':
            self.wl_base = CustomVertexHistogram, {'sparse': False}
        elif gkernel == 'wloa':
            self.wl_base = CustomVertexHistogram, {'sparse': False, 'oa': True}
        else:
            raise NotImplementedError(gkernel + ' is not a valid graph kernel choice!')

        self.gkernel = None
        # only applicable for the DARTS search space, where we optimise two graphs jointly.
        self.gkernel_reduce = None

        # sometimes (especially for NAS-Bench-201), we can have invalid graphs with all nodes being pruned. Remove
        # these graphs at training time.
        if self.space == 'nasbench301' or self.space == 'darts':
            # For NAS-Bench-301 or DARTS search space, we need to search for 2 cells (normal and reduction simultaneously)
            valid_indices = [i for i in range(len(xtrain[0])) if len(xtrain[0][i]) and len(xtrain[1][i])]
            self.x = np.array(xtrain)[:, valid_indices]
            # self.x = [xtrain[i] for i in valid_indices]
            self.xtrain_converted = [list(graph_from_networkx(self.x[0], node_label, )),
                                     list(graph_from_networkx(self.x[1], node_label, )), ]

        else:
            valid_indices = [i for i in range(len(xtrain)) if len(xtrain[i])]
            self.x = np.array([xtrain[i] for i in valid_indices])
            self.xtrain_converted = list(graph_from_networkx(self.x, node_label, ))

        ytrain = np.array(ytrain)[valid_indices]
        self.y_ = deepcopy(torch.tensor(ytrain, dtype=torch.float32), )
        self.y, self.y_mean, self.y_std = _normalize(deepcopy(self.y_))
        # number of steps of training
        self.num_steps = num_steps

        # other hyperparameters
        self.max_noise_var = max_noise_var
        self.max_h = max_h
        self.optimize_noise_var = optimize_noise_var

        self.node_label = node_label
        self.K_i = None

    def forward(self, Xnew, full_cov=False, ):

        if self.K_i is None:
            raise ValueError("The GraphGP model has not been fit!")

        # At testing time, similarly we first inspect to see whether there are invalid graphs
        if self.space == 'nasbench301' or self.space == 'darts':
            invalid_indices = [i for i in range(len(Xnew[0])) if len(Xnew[0][i]) == 0 or len(Xnew[1][i]) == 0]
        else:
            nnodes = np.array([len(x) for x in Xnew])
            invalid_indices = np.argwhere(nnodes == 0)

        # replace the invalid indices with something valid
        patience = 100
        for i in range(len(Xnew)):
            if i in invalid_indices:
                patience -= 1
                continue
            break
        if patience < 0:
            # All architectures are invalid!
            return torch.zeros(len(Xnew)), torch.zeros(len(Xnew))
        for j in invalid_indices:
            if self.space == 'nasbench301' or self.space == 'darts':
                Xnew[0][int(j)] = Xnew[0][i]
                Xnew[1][int(j)] = Xnew[1][i]
            else:
                Xnew[int(j)] = Xnew[i]

        if self.space == 'nasbench301' or self.space == 'darts':
            Xnew_T = np.array(Xnew)
            Xnew = np.array(
                [list(graph_from_networkx(Xnew_T[0], self.node_label, )),
                 list(graph_from_networkx(Xnew_T[1], self.node_label, )),
                 ])

            X_full = np.concatenate((np.array(self.xtrain_converted), Xnew), axis=1)
            K_full = torch.tensor(
                0.5 * torch.tensor(self.gkernel.fit_transform(X_full[0]), dtype=torch.float32) +
                0.5 * torch.tensor(self.gkernel_reduce.fit_transform(X_full[1]), dtype=torch.float32)
            )
            # Kriging equations
            K_s = K_full[:len(self.x[0]):, len(self.x[0]):]
            K_ss = K_full[len(self.x[0]):, len(self.x[0]):] + self.likelihood * torch.eye(Xnew.shape[1], )
        else:
            Xnew = list(graph_from_networkx(Xnew, self.node_label, ))
            X_full = self.xtrain_converted + Xnew
            K_full = torch.tensor(self.gkernel.fit_transform(X_full), dtype=torch.float32)
            # Kriging equations
            K_s = K_full[:len(self.x):, len(self.x):]
            K_ss = K_full[len(self.x):, len(self.x):] + self.likelihood * torch.eye(len(Xnew), )
        mu_s = K_s.t()@self.K_i@self.y
        cov_s = K_ss - K_s.t()@self.K_i@K_s
        cov_s = torch.clamp(cov_s, self.likelihood, np.inf)
        mu_s = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s = torch.sqrt(cov_s)
        std_s = unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s ** 2
        if not full_cov:
            cov_s = torch.diag(cov_s)
        # replace the invalid architectures with zeros
        mu_s[torch.tensor(invalid_indices, dtype=torch.long)] = torch.tensor(0., dtype=torch.float32)
        cov_s[torch.tensor(invalid_indices, dtype=torch.long)] = torch.tensor(0., dtype=torch.float32)
        return mu_s, cov_s

    def fit(self):

        xtrain_grakel = self.xtrain_converted
        # Valid values of h are non-negative integers. Here we test each of them once, and pick the one that leads to
        # the highest marginal likelihood of the GP model.
        if self.h == 'auto':
            best_nlml = torch.tensor(np.inf, dtype=torch.float32)
            best_h = None
            best_K = None
            for candidate in [h for h in range(self.max_h + 1)]:
                gkernel = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=candidate)
                if self.space == 'nasbench301' or self.space == 'darts':
                    gkernel_reduce = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=candidate)
                    K = (torch.tensor(gkernel.fit_transform(xtrain_grakel[0], self.y), dtype=torch.float32)
                         + torch.tensor(gkernel_reduce.fit_transform(xtrain_grakel[1], self.y), dtype=torch.float32)) \
                        / 2.
                else:
                    K = torch.tensor(gkernel.fit_transform(xtrain_grakel, self.y), dtype=torch.float32)
                K_i, logDetK = _compute_pd_inverse(K, self.likelihood)
                nlml = -_compute_log_marginal_likelihood(K_i, logDetK, self.y)
                if nlml < best_nlml:
                    best_nlml = nlml
                    best_h = candidate
                    best_K = torch.clone(K)
            K = best_K
            self.gkernel = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=best_h)
            self.gkernel_reduce = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=best_h)
        else:
            self.gkernel = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=self.h)
            self.gkernel_reduce = WeisfeilerLehman(base_graph_kernel=self.wl_base, h=self.h)
            if self.space == 'nasbench301' or self.space == 'darts':
                K = (torch.tensor(self.gkernel.fit_transform(xtrain_grakel[0], self.y), dtype=torch.float32)
                     + torch.tensor(self.gkernel_reduce.fit_transform(xtrain_grakel[1], self.y), dtype=torch.float32)) \
                    / 2.
            else:
                K = torch.tensor(self.gkernel.fit_transform(xtrain_grakel, self.y), dtype=torch.float32)

        # CONDITIONAL on the valid h parameter picked, here we optimise the noise as a hyperparameter using standard
        # gradient-based optimisation. Here by default we use Adam optimizer.
        if self.optimize_noise_var:
            likelihood = torch.tensor(self.likelihood, dtype=torch.float32, requires_grad=True)
            optim = torch.optim.Adam([likelihood], lr=0.1)
            for i in range(self.num_steps):
                optim.zero_grad()
                K_i, logDetK = _compute_pd_inverse(K, likelihood)
                nlml = -_compute_log_marginal_likelihood(K_i, logDetK, self.y)
                nlml.backward()
                optim.step()
                with torch.no_grad():
                    likelihood.clamp_(1e-7, self.max_noise_var)
            # finally
            K_i, logDetK = _compute_pd_inverse(K, likelihood)
            self.K_i = K_i.detach().cpu()
            self.logDetK = logDetK.detach().cpu()
            self.likelihood = likelihood.item()
        else:
            # Compute the inverse covariance matrix
            self.K_i, self.logDetK = _compute_pd_inverse(K, self.likelihood)


class GPWLPredictor(BaseGPModel):

    def __init__(self, kernel_type='wloa', ss_type='nasbench201', optimize_gp_hyper=False,
                 h=2, num_steps=200):
        super(GPWLPredictor, self).__init__(None, ss_type, kernel_type, optimize_gp_hyper)
        self.h = h
        self.num_steps = num_steps
        self.need_separate_hpo = True
        self.model = None

    def _convert_data(self, data: list):
        if self.ss_type == 'nasbench101':
            converted_data = [convert_n101_arch_to_graph(arch) for arch in data]
        elif self.ss_type == 'nasbench201':
            converted_data = [convert_n201_arch_to_graph(arch) for arch in data]
        elif self.ss_type == 'nasbench301' or self.ss_type == 'darts':
            converted_data = [convert_darts_arch_to_graph(arch) for arch in data]
            # the converted data is in shape of (N, 2). Transpose to (2,N) for convenience later on.
            converted_data = np.array(converted_data).T.tolist()
        else:
            raise NotImplementedError("Search space %s is not implemented!" % self.ss_type)
        return converted_data

    def get_model(self, train_data, **kwargs):
        X_train, y_train = train_data
        # log-transform
        y_train = _transform(y_train)
        # first convert data to networkx
        X_train = self._convert_data(X_train)
        self.model = GraphGP(X_train, y_train, self.kernel_type, h=self.h, num_steps=self.num_steps,
                             optimize_noise_var=self.optimize_gp_hyper, space=self.ss_type)
        # fit the model
        self.model.fit()
        return self.model

    def predict(self, input_data, **kwargs):
        X_test = self._convert_data(input_data)
        mean, cov = self.model.forward(X_test, full_cov=True)
        mean = mean.cpu().detach().numpy()
        mean = _untransform(mean)
        return mean

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        # if not isinstance(xtrain[0], nx.DiGraph):
        xtrain_conv = self._convert_data(xtrain)
        ytrain_transformed = _transform(ytrain)

        self.model = GraphGP(xtrain_conv, ytrain_transformed, self.kernel_type, h=self.h, num_steps=self.num_steps,
                             optimize_noise_var=self.optimize_gp_hyper, space=self.ss_type)
        # fit the model
        self.model.fit()
        print('Finished fitting GP')
        # predict on training data for checking
        train_pred = self.query(xtrain).squeeze()
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None):
        """alias for predict"""
        return self.predict(xtest)