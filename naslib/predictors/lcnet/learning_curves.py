# This code is from https://github.com/automl/pybnn
# pybnn authors: Aaron Klein, Moritz Freidank

import emcee
import logging
import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm

from naslib.predictors.lcnet.curvefunctions import curve_combination_models, \
    model_defaults, all_models
from naslib.predictors.lcnet.curvemodels import MLCurveModel


def recency_weights(num):
    if num == 1:
        return np.ones(1)
    else:
        recency_weights = [10 ** (1. / num)] * num
        recency_weights = recency_weights ** (np.arange(0, num))
        return recency_weights


def model_ln_prob(theta, model, x, y):
    return model.ln_prob(theta, x, y)


class MCMCCurveModelCombination(object):
    def __init__(self,
                 xlim,
                 ml_curve_models=None,
                 burn_in=500,
                 nwalkers=100,
                 nsamples=2500,
                 normalize_weights=True,
                 monotonicity_constraint=True,
                 soft_monotonicity_constraint=False,
                 initial_model_weight_ml_estimate=False,
                 normalized_weights_initialization="constant",
                 strictly_positive_weights=True,
                 sanity_check_prior=True,
                 nthreads=1,
                 recency_weighting=True):
        """
            xlim: the point on the x axis we eventually want to make predictions for.
        """
        if ml_curve_models is None:
            curve_models = []
            for model_name in curve_combination_models:
                if model_name in model_defaults:
                    m = MLCurveModel(function=all_models[model_name],
                                     default_vals=model_defaults[model_name],
                                     recency_weighting=False)
                else:
                    m = MLCurveModel(function=all_models[model_name],
                                     recency_weighting=False)
                curve_models.append(m)
            self.ml_curve_models = curve_models
        else:
            self.ml_curve_models = ml_curve_models

        self.xlim = xlim
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.normalize_weights = normalize_weights
        assert not (
                monotonicity_constraint and soft_monotonicity_constraint), "choose either the monotonicity_constraint or the soft_monotonicity_constraint, but not both"
        self.monotonicity_constraint = monotonicity_constraint
        self.soft_monotonicity_constraint = soft_monotonicity_constraint
        self.initial_model_weight_ml_estimate = initial_model_weight_ml_estimate
        self.normalized_weights_initialization = normalized_weights_initialization
        self.strictly_positive_weights = strictly_positive_weights
        self.sanity_check_prior = sanity_check_prior
        self.nthreads = nthreads
        self.recency_weighting = recency_weighting
        # the constant used for initializing the parameters in a ball around the ML parameters
        self.rand_init_ball = 1e-6
        self.name = "model combination"  # (%s)" % ", ".join([model.name for model in self.ml_curve_models])

        if self.monotonicity_constraint:
            self._x_mon = np.linspace(2, self.xlim, 50)
        else:
            self._x_mon = np.asarray([2, self.xlim])

        # TODO check that burnin is lower than nsamples

    def fit(self, x, y, model_weights=None):
        if self.fit_ml_individual(x, y, model_weights):
            # run MCMC:
            logging.info('Fitted models!')
            self.fit_mcmc(x, y)
            logging.info('Fitted mcmc!')
            return True
        else:
            logging.warning("fit_ml_individual failed")
            return False

    def y_lim_sanity_check(self, ylim):
        # just make sure that the prediction is not below 0 nor insanely big
        # HOWEVER: there might be cases where some models might predict value larger than 1.0
        # and this is alright, because in those cases we don't necessarily want to stop a run.
        assert not isinstance(ylim, np.ndarray)
        if not np.isfinite(ylim) or ylim < 0. or ylim > 100.0:
            return False
        else:
            return True

    def y_lim_sanity_check_array(self, ylim):
        # just make sure that the prediction is not below 0 nor insanely big
        # HOWEVER: there might be cases where some models might predict value larger than 1.0
        # and this is alright, because in those cases we don't necessarily want to stop a run.
        assert isinstance(ylim, np.ndarray)
        return ~(~np.isfinite(ylim) | (ylim < 0.) | (ylim > 100.0))

    def fit_ml_individual(self, x, y, model_weights):
        """
            Do a ML fit for each model individually and then another ML fit for the combination of models.
        """
        self.fit_models = []
        for model in self.ml_curve_models:
            if model.fit(x, y):
                ylim = model.predict(self.xlim)
                if not self.y_lim_sanity_check(ylim):
                    print("ML fit of model %s is out of bound range [0.0, "
                          "100.] at xlim." % (model.function.__name__))
                    continue
                params, sigma = model.split_theta_to_array(model.ml_params)
                if not np.isfinite(self._ln_model_prior(model, np.array([params]))[0]):
                    print("ML fit of model %s is not supported by prior." %
                          model.function.__name__)
                    continue
                self.fit_models.append(model)

        if len(self.fit_models) == 0:
            return False

        if model_weights is None:
            if self.normalize_weights:
                if self.normalized_weights_initialization == "constant":
                    # initialize with a constant value
                    # we will sample in this unnormalized space and then later normalize
                    model_weights = [10. for model in self.fit_models]
                else:  # self.normalized_weights_initialization == "normalized"
                    model_weights = [1. / len(self.fit_models) for model in
                                     self.fit_models]
            else:
                if self.initial_model_weight_ml_estimate:
                    model_weights = self.get_ml_model_weights(x, y)
                    non_zero_fit_models = []
                    non_zero_weights = []
                    for w, model in zip(model_weights, self.fit_models):
                        if w > 1e-4:
                            non_zero_fit_models.append(model)
                            non_zero_weights.append(w)
                    self.fit_models = non_zero_fit_models
                    model_weights = non_zero_weights
                else:
                    model_weights = [1. / len(self.fit_models) for model in
                                     self.fit_models]

        # build joint ml estimated parameter vector
        model_params = []
        all_model_params = []
        for model in self.fit_models:
            params, sigma = model.split_theta_to_array(model.ml_params)
            model_params.append(params)
            all_model_params.extend(params)

        y_predicted = self._predict_given_params(
            x, [np.array([mp]) for mp in model_params],
            np.array([model_weights]))
        sigma = (y - y_predicted).std()

        self.ml_params = self._join_theta(all_model_params, sigma, model_weights)
        self.ndim = len(self.ml_params)
        if self.nwalkers < 2 * self.ndim:
            self.nwalkers = 2 * self.ndim
            logging.warning("increasing number of walkers to 2*ndim=%d" % (
                self.nwalkers))
        return True

    def get_ml_model_weights(self, x, y_target):
        """
            Get the ML estimate of the model weights.
        """

        """
            Take all the models that have been fit using ML.
            For each model we get a prediction of y: y_i

            Now how can we combine those to reduce the squared error:

                argmin_w (y_target - w_1 * y_1 - w_2 * y_2 - w_3 * y_3 ...)

            Deriving and setting to zero we get a linear system of equations that we need to solve.


            Resource on QP:
            http://stats.stackexchange.com/questions/21565/how-do-i-fit-a-constrained-regression-in-r-so-that-coefficients-total-1
            http://maggotroot.blogspot.de/2013/11/constrained-linear-least-squares-in.html
        """
        num_models = len(self.fit_models)
        y_predicted = []
        b = []
        for model in self.fit_models:
            y_model = model.predict(x)
            y_predicted.append(y_model)
            b.append(y_model.dot(y_target))
        a = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                a[i, j] = y_predicted[i].dot(y_predicted[j])
                # if i == j:
                #    a[i, j] -= 0.1 #constraint the weights!
        a_rank = np.linalg.matrix_rank(a)
        if a_rank != num_models:
            print("Rank %d not sufficcient for solving the linear system. %d "
                  "needed at least." % (a_rank, num_models))
        try:
            print(np.linalg.lstsq(a, b)[0])
            print(np.linalg.solve(a, b))
            print(nnls(a, b)[0])
            ##return np.linalg.solve(a, b)
            weights = nnls(a, b)[0]
            # weights = [w if w > 1e-4 else 1e-4 for w in weights]
            return weights
        # except LinAlgError as e:
        except:
            return [1. / len(self.fit_models) for model in self.fit_models]

    # priors
    def _ln_prior(self, theta):
        # TODO remove this check, accept only 2d data
        if len(theta.shape) == 1:
            theta = theta.reshape((1, -1))

        ln = np.array([0.] * len(theta))
        model_params, sigma, model_weights = self._split_theta(theta)

        # we expect all weights to be positive
        # TODO add unit test for this!

        if self.strictly_positive_weights:
            violation = np.any(model_weights < 0, axis=1)
            ln[violation] = -np.inf

        for model, params in zip(self.fit_models, model_params):
            # Only calculate the prior further when the value is still finite
            mask = np.isfinite(ln)
            if np.sum(mask) == 0:
                break
            ln[mask] += self._ln_model_prior(model, params[mask])

            # if self.normalize_weights:
            # when we normalize we expect all weights to be positive
        return ln

    def _ln_model_prior(self, model, params):
        prior = np.array([0.0] * len(params))
        # reshaped_params = [
        #    np.array([params[j][i]
        #              for j in range(len(params))]).reshape((-1, 1))
        #    for i in range(len(params[0]))]
        reshaped_params = [params[:, i].reshape((-1, 1))
                           for i in range(len(params[0]))]

        # prior_stats = []
        # prior_stats.append((0, np.mean(~np.isfinite(prior))))

        # TODO curvefunctions must be vectorized, too
        # y_mon = np.array([model.function(self._x_mon, *params_)
        #                  for params_ in params])

        # Check, is this predict the most expensive part of the whole code? TODO
        # y_mon = model.function(self._x_mon, *reshaped_params)

        if self.monotonicity_constraint:
            y_mon = model.function(self._x_mon, *reshaped_params)
            # check for monotonicity(this obviously this is a hack, but it works for now):
            constraint_violated = np.any(np.diff(y_mon, axis=1) < 0, axis=1)
            prior[constraint_violated] = -np.inf
            # for i in range(len(y_mon)):
            #    if np.any(np.diff(y_mon[i]) < 0):
            #        prior[i] = -np.inf

        elif self.soft_monotonicity_constraint:
            y_mon = model.function(self._x_mon[[0, -1]], *reshaped_params)
            # soft monotonicity: defined as the last value being bigger than the first one
            not_monotone = [y_mon[i, 0] > y_mon[i, -1] for i in range(len(y_mon))]
            if any(not_monotone):
                for i, nm in enumerate(not_monotone):
                    if nm:
                        prior[i] = -np.inf

        else:
            y_mon = model.function(self._x_mon, *reshaped_params)

        # TODO curvefunctions must be vectorized, too
        # ylim = np.array([model.function(self.xlim, *params_)
        #                for params_ in params])
        # ylim = model.function(self.xlim, *reshaped_params)
        ylim = y_mon[:, -1]

        # sanity check for ylim
        if self.sanity_check_prior:
            sane = self.y_lim_sanity_check_array(ylim)
            prior[~sane.flatten()] = -np.inf
            # for i, s in enumerate(sane):
            #    if not s:
            #        prior[i] = -np.inf

        # TODO vectorize this!
        mask = np.isfinite(prior)
        for i, params_ in enumerate(params):
            # Only check parameters which are not yet rejected
            if mask[i] and not model.are_params_in_bounds(params_):
                prior[i] = -np.inf

        # prior_stats.append((3, np.mean(~np.isfinite(prior))))
        # print(prior_stats)
        return prior

    # likelihood
    def _ln_likelihood(self, theta, x, y):
        y_model, sigma = self._predict_given_theta(x, theta)
        n_models = len(y_model)

        if self.recency_weighting:
            raise NotImplementedError()
            weight = recency_weights(len(y))
            ln_likelihood = (
                    weight * norm.logpdf(y - y_model, loc=0, scale=sigma)).sum()
        else:
            # ln_likelihood = [norm.logpdf(y - y_model_, loc=0, scale=sigma_).sum()
            #                 for y_model_, sigma_ in zip(y_model, sigma)]
            # ln_likelihood = np.array(ln_likelihood)
            loc = np.zeros((n_models, 1))
            sigma = sigma.reshape((-1, 1))
            ln_likelihood2 = norm.logpdf(y - y_model, loc=loc,
                                         scale=sigma).sum(axis=1)
            # print(ln_likelihood == ln_likelihood2)
            ln_likelihood = ln_likelihood2

        ln_likelihood[~np.isfinite(ln_likelihood)] = -np.inf
        return ln_likelihood

    def _ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self._ln_prior(theta)
        lp[~np.isfinite(lp)] = -np.inf
        ln_prob = lp + self._ln_likelihood(theta, x, y)
        return ln_prob

    def _split_theta(self, theta):
        """
            theta is structured as follows:
            for each model i
                for each model parameter j
            theta = (theta_ij, sigma, w_i)
        """
        # TODO remove this check, theta should always be 2d!
        if len(theta.shape) == 1:
            theta = theta.reshape((1, -1))

        all_model_params = []
        for model in self.fit_models:
            num_model_params = len(model.function_params)
            model_params = theta[:, :num_model_params]
            all_model_params.append(model_params)

            theta = theta[:, num_model_params:]

        sigma = theta[:, 0]
        model_weights = theta[:, 1:]
        assert model_weights.shape[1] == len(self.fit_models)
        return all_model_params, sigma, model_weights

    def _join_theta(self, model_params, sigma, model_weights):
        # assert len(model_params) == len(model_weights)
        theta = []
        theta.extend(model_params)
        theta.append(sigma)
        theta.extend(model_weights)
        return theta

    def fit_mcmc(self, x, y):
        # initialize in an area around the starting position

        class PseudoPool(object):
            def map(self, func, proposals):
                return [f for f in func(np.array(proposals))]

        rstate0 = np.random.RandomState(1)
        assert self.ml_params is not None
        pos = [self.ml_params + self.rand_init_ball * rstate0.randn(self.ndim)
               for i in range(self.nwalkers)]

        if self.nthreads <= 1:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                                            self.ndim,
                                            self._ln_prob,
                                            args=(x, y),
                                            pool=PseudoPool())
        else:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                model_ln_prob,
                args=(self, x, y),
                threads=self.nthreads)
        sampler.run_mcmc(pos, self.nsamples, rstate0=rstate0)
        self.mcmc_chain = sampler.chain

        if self.normalize_weights:
            self.normalize_chain_model_weights()

    def normalize_chain_model_weights(self):
        """
            In the chain we sample w_1,... w_i however we are interested in the model
            probabilities p_1,... p_i
        """
        model_weights_chain = self.mcmc_chain[:, :, -len(self.fit_models):]
        model_probabilities_chain = model_weights_chain / model_weights_chain.sum(
            axis=2)[:, :, np.newaxis]
        # replace in chain
        self.mcmc_chain[:, :,
        -len(self.fit_models):] = model_probabilities_chain

    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def print_probs(self):
        burned_in_chain = self.get_burned_in_samples()
        model_probabilities = burned_in_chain[:, -len(self.fit_models):]
        print(model_probabilities.mean(axis=0))

    def _predict_given_theta(self, x, theta):
        """
            returns y_predicted, sigma
        """
        model_params, sigma, model_weights = self._split_theta(theta)

        y_predicted = self._predict_given_params(x, model_params, model_weights)

        return y_predicted, sigma

    def _predict_given_params(self, x, model_params, model_weights):
        """
            returns y_predicted
        """

        if self.normalize_weights:
            model_weight_sum = np.sum(model_weights, axis=1)
            model_ws = (model_weights.transpose() / model_weight_sum).transpose()
        else:
            model_ws = model_weights

        # # TODO vectorize!
        # vectorized_predictions = []
        # for i in range(len(model_weights)):
        #     y_model = []
        #     for model, model_w, params in zip(self.fit_models, model_ws[i],
        #                                       model_params):
        #         y_model.append(model_w * model.function(x, *params[i]))
        #     y_predicted = functools.reduce(lambda a, b: a + b, y_model)
        #     vectorized_predictions.append(y_predicted)

        len_x = len(x) if hasattr(x, '__len__') else 1
        test_predictions = np.zeros((len(model_weights), len_x))
        for model, model_w, params in zip(self.fit_models, model_ws.transpose(),
                                          model_params):
            params2 = [params[:, i].reshape((-1, 1))
                       for i in range(params.shape[1])]
            params = params2
            # params = [np.array([params[j][i] for j in range(len(params))]).reshape((-1, 1))
            #          for i in range(len(params[0]))]
            # print('Diff', np.sum(np.array(params2)
            #             - np.array(params).reshape((len(params2), -1))))
            prediction = model_w.reshape((-1, 1)) * model.function(x, *params)
            test_predictions += prediction

        return test_predictions
        # return np.array(vectorized_predictions)

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int), (x, type(x))

        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            model_params, sigma, model_weights = self._split_theta(theta)
            y_predicted = self._predict_given_params(x, model_params,
                                                     model_weights)
            predictions.append(y_predicted)
        return np.asarray(predictions)

    def prob_x_greater_than(self, x, y, theta):
        """
            P(f(x) > y | Data, theta)
        """
        model_params, sigma, model_weights = self._split_theta(theta)

        y_predicted = self._predict_given_params(x, model_params, model_weights)

        cdf = norm.cdf(y, loc=y_predicted, scale=sigma)

        return 1. - cdf

    def posterior_prob_x_greater_than(self, x, y, thin=1):
        """
            P(f(x) > y | Data)

            Posterior probability that f(x) is greater than y.
        """
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:
            probs.append(self.prob_x_greater_than(x, y, theta))

        return np.ma.masked_invalid(probs).mean()
