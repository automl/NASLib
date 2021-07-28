# This code is from https://github.com/automl/pylearningcurvepredictor
# pylearningcurvepredictor author: Tobias Domhan, tdomhan

import numpy as np
from scipy.stats import norm
import time
from typing import List

from naslib.predictors.lce.parametric_model import ParametricModel


class ParametricEnsemble:
    def __init__(self, parametric_models: List[ParametricModel]):
        self.parametric_models = parametric_models
        self.weights = [1 / len(parametric_models)] * len(parametric_models)

    def fit(self, x, fit_weights=False):
        if fit_weights:
            raise NotImplementedError
        else:
            self.x = x
            for model in self.parametric_models:
                model.fit(x)
            self.params = {
                model.name: model.get_params() for model in self.parametric_models
            }
            # set sigma squared to be the sample variance
            sum_sq = 0
            for i in range(x.shape[0]):
                sum_sq += (x[i] - self.predict(i + 1)) ** 2
            self.sigma_sq = (1 / x.shape[0]) * sum_sq

    def predict(self, x, params=None, weights=None):
        if params is not None:
            return sum(
                [
                    w * model.predict(x, params=params[model.name])
                    for (w, model) in zip(weights, self.parametric_models)
                ]
            )
        return sum(
            [
                w * model.predict(x)
                for (w, model) in zip(self.weights, self.parametric_models)
            ]
        )

    def get_params(self):
        return self.params, self.weights, self.sigma_sq

    def set_params(self, params, sigma_sq=None):
        self.params = params
        if sigma_sq:
            self.sigma_sq = sigma_sq

    def perturb_params(self, params, weights, sigma_sq, var):

        # free variables: sigma squared, weights, model parameters
        deg_freedom = (
            1
            + len(self.weights)
            + sum([model.degrees_freedom for model in self.parametric_models])
        )
        perturbation = np.random.normal(loc=0, scale=var, size=(deg_freedom,))
        perturbed_params = params.copy()
        perturbed_weights = weights.copy()
        pos = 0
        for model in self.parametric_models:
            perturbed_params[model.name] += np.concatenate(
                [perturbation[pos : pos + model.degrees_freedom], np.zeros((1,))]
            )
            pos += model.degrees_freedom
        for i in range(len(self.weights)):
            perturbed_weights[i] += perturbation[pos]
            pos += 1
        perturbed_sigma_sq = sigma_sq + perturbation[-1]
        return perturbed_params, perturbed_weights, perturbed_sigma_sq

    def mcmc(self, x, N=10000, var=0.0001, fit_weights=False, verbose=False):
        (
            acceptances,
            stochastic_rejections,
            pathological_rejections,
            way_off_rejections,
        ) = (0, 0, 0, 0)

        self.fit(x, fit_weights)  # initialize with mle estimates for each model

        curvelen = x.shape[0]
        start = time.time()
        params = self.params.copy()
        weights = self.weights.copy()
        sigma_sq = self.sigma_sq
        self.mcmc_sample_params = []

        zero_likelihood = False

        # sampling loop
        for t in range(N):
            self.mcmc_sample_params.append((params, weights))

            if verbose:
                if t == 1:
                    last_power_two = t
                elif t == 2 * last_power_two:
                    last_power_two = t
                    print(
                        f"Completed {t} Metropolis steps in {time.time() - start} seconds."
                    )

            current_log_likelihood = 0
            for j in range(curvelen):
                jth_error = self.predict(j + 1, params=params, weights=weights) - x[j]
                point_likelihood = norm.pdf(jth_error, scale=np.sqrt(sigma_sq))
                if not point_likelihood > 0:
                    point_likelihood = 1e-10
                    if not zero_likelihood:
                        zero_likelihood = True
                        print("point likelihood was 0")
                current_log_likelihood += np.log(point_likelihood)

            (
                candidate_params,
                candidate_weights,
                candidate_sigma_sq,
            ) = self.perturb_params(params, weights, sigma_sq, var)
            if candidate_sigma_sq <= 0:
                # reject, sigma squared must be positive
                continue
            candidate_log_likelihood = 0
            min_point_likelihood = 1
            for j in range(curvelen):
                jth_error = (
                    self.predict(
                        j + 1, params=candidate_params, weights=candidate_weights
                    )
                    - x[j]
                )
                point_likelihood = norm.pdf(
                    jth_error, scale=np.sqrt(candidate_sigma_sq)
                )
                min_point_likelihood = min(min_point_likelihood, point_likelihood)
                if point_likelihood > 0:
                    candidate_log_likelihood += np.log(point_likelihood)
            if min_point_likelihood == 0:
                # reject due to vanishing point likelihood
                continue

            acceptance_probability = min(
                1, np.exp(candidate_log_likelihood - current_log_likelihood)
            )
            if self.predict(
                curvelen + 1, params=candidate_params, weights=candidate_weights
            ) > self.predict(1, params=candidate_params, weights=candidate_weights):
                if np.random.random() < acceptance_probability:
                    params = candidate_params
                    weights = candidate_weights
                    sigma_sq = candidate_sigma_sq
                    acceptances += 1

        print(
            f"Completed with acceptance rate {acceptances / N} in {time.time() - start} seconds."
        )

    def mcmc_sample_predict(self, x):
        return sum(
            [
                self.predict(x, params=p[0], weights=p[1])
                for p in self.mcmc_sample_params
            ]
        ) / len(self.mcmc_sample_params)

    def mcmc_sample_eval(self, epochs, y):
        predictions = self.mcmc_sample_predict(epochs)
        mse = 0
        for i in range(y.shape[0]):
            print("pred", predictions[i], "real", y[i])
            mse += (predictions[i] - y[i]) ** 2
            print("mse", (predictions[i] - y[i]) ** 2)
        mse /= y.shape[0]
        return mse
