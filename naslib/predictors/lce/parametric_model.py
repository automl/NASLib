import numpy as np
from scipy.optimize import minimize

def optimize_model_class(fn_class, deg_freedom, x, y, bounds=None):
    def likelihood(parameters):
        return (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(parameters[-1] ** 2) + 1 /
            (2 * parameters[-1] ** 2) * sum((y - fn_class(parameters)(x)) ** 2))
    opt_model = minimize(likelihood, np.array([1] * (deg_freedom + 1)), method='L-BFGS-B', bounds=bounds)
    return opt_model['x']


class ParametricModel:
    def __init__(self, model_class, degrees_freedom, name, bounds=None):
        self.model_class = model_class
        self.degrees_freedom = degrees_freedom
        self.name = name
        self.bounds = bounds

    def fit(self, x):
        # x is np array dim 1
        self.x = x
        self.params = optimize_model_class(self.model_class, self.degrees_freedom,
                                           list(range(1, x.shape[0] + 1)), x, bounds=self.bounds)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def predict(self, x, params=None):
        if params is not None:
            return self.model_class(params)(x)
        return self.model_class(self.params)(x)

def parametric_model(func):
    return (lambda params: np.vectorize(lambda x: func(params, x)))

# usage for parametric_model decorated function "fn":
# fn(params)(argument) where argument can be numeric, list, or np.ndarray

@parametric_model
def vapor_pressure(params, x):
    a, b, c, _ = params
    return a + b/x + c * np.log(x)

@parametric_model
def pow3(params, x):
    a, c, alpha, _ = params
    return c - a * ((1. * x) ** -alpha)

@parametric_model
def logloglinear(params, x):
    a, b, _ = params
    return np.log(a * np.log(x) + b)

@parametric_model
def hill3(params, x):
    ymax, eta, kappa, _ = params
    return ymax * (x ** eta) / (kappa * eta + x ** eta)

@parametric_model
def logpower(params, x):
    a, b, c, _ = params
    return a / (1 + (x / np.exp(b)) ** c)

@parametric_model
def pow4(params, x):
    a, b, c, alpha, _ = params
    return c - (a * x + b) ** (-1 * alpha)

@parametric_model
def mmf(params, x):
    alpha, beta, delta, kappa, _ = params
    return alpha - (alpha - beta) / (1 + (kappa * x) ** delta)

@parametric_model
def exp4(params, x):
    a, b, c, alpha, _ = params
    return c - np.exp(- a * (x ** alpha) + b)

@parametric_model
def janoschek(params, x):
    alpha, beta, kappa, delta, _ = params
    return alpha - (alpha - beta) * np.exp(- kappa * (x ** delta))

@parametric_model
def weibull(params, x):
    alpha, beta, kappa, delta, _ = params
    return alpha - (alpha - beta) * np.exp(- (kappa * x) ** delta)

@parametric_model
def ilog2(params, x):
    a, c, _ = params
    return c - (a / np.log(x + 1))

model_name_list = [
    'vapor_pressure',
    'pow3',
    'logloglinear',
    'logpower',
    'pow4',
    'mmf',
    'exp4',
    'janoschek',
    'weibull',
    'ilog2',
    'hill3'
]

positive_only = (np.finfo(float).eps, np.inf)
no_bound = (-np.inf, np.inf)

model_config = {}

model_config['vapor_pressure'] = {
    'model': vapor_pressure,
    'deg_freedom': 3,
    'bounds': None
}

model_config['pow3'] = {
    'model': pow3,
    'deg_freedom': 3,
    'bounds': ((0, 2), positive_only, positive_only, positive_only),
}

model_config['logloglinear'] = {
    'model': logloglinear,
    'deg_freedom': 2,
    'bounds': (positive_only,) * 3,
}

model_config['logpower'] = {
    'model': logpower,
    'deg_freedom': 3,
    'bounds': None
}

model_config['pow4'] = {
    'model': pow4,
    'deg_freedom': 4,
    'bounds': (positive_only,) * 5,
}

model_config['mmf'] = {
    'model': mmf,
    'deg_freedom': 4,
    'bounds': None
}

model_config['exp4'] = {
    'model': exp4,
    'deg_freedom': 4,
    'bounds': None
}

model_config['janoschek'] = {
    'model': janoschek,
    'deg_freedom': 4,
    'bounds': None
}

model_config['weibull'] = {
    'model': weibull,
    'deg_freedom': 4,
    'bounds': (no_bound, no_bound, positive_only, no_bound, positive_only),
}

model_config['ilog2'] = {
    'model': ilog2,
    'deg_freedom': 2,
    'bounds': (positive_only, positive_only, positive_only),
}

model_config['hill3'] = {
    'model': hill3,
    'deg_freedom': 3,
    'bounds': (positive_only, no_bound, positive_only, positive_only),
}

def construct_parametric_model(model_config, model_name):
    return ParametricModel(model_config[model_name]['model'],
        model_config[model_name]['deg_freedom'],
        model_name,
        model_config[model_name]['bounds'])
