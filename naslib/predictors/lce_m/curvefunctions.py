# This code is from https://github.com/automl/pybnn
# pybnn authors: Aaron Klein, Moritz Freidank

import numpy as np

# all the models that we considered at some point
all_models = {}
model_defaults = {}
display_name_mapping = {}


def pow3(x, c, a, alpha):
    return c - a * x ** (-alpha)


all_models["pow3"] = pow3
model_defaults["pow3"] = {"c": 0.84, "a": 0.52, "alpha": 0.01}
display_name_mapping["pow3"] = "pow$_3$"


def linear(x, a, b):
    return a * x + b


# models["linear"] = linear
all_models["linear"] = linear

"""
    Source: curve expert
"""


def log_power(x, a, b, c):
    # logistic power
    return a / (1.0 + (x / np.exp(b)) ** c)


all_models["log_power"] = log_power
model_defaults["log_power"] = {"a": 0.77, "c": -0.51, "b": 2.98}
display_name_mapping["log_power"] = "log power"


def weibull(x, alpha, beta, kappa, delta):
    """
    Weibull modell

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-ordinate for the point of inflection
    """
    return alpha - (alpha - beta) * np.exp(-((kappa * x) ** delta))


all_models["weibull"] = weibull
model_defaults["weibull"] = {"alpha": 0.7, "beta": 0.1, "kappa": 0.01, "delta": 1}
display_name_mapping["weibull"] = "Weibull"


def mmf(x, alpha, beta, kappa, delta):
    """
    Morgan-Mercer-Flodin

    description:
    Nonlinear Regression page 342
    http://bit.ly/1jodG17
    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    kappa: growth rate
    beta: initial value
    delta: controls the point of inflection
    """
    return alpha - (alpha - beta) / (1.0 + (kappa * x) ** delta)


all_models["mmf"] = mmf
model_defaults["mmf"] = {"alpha": 0.7, "kappa": 0.01, "beta": 0.1, "delta": 5}
display_name_mapping["mmf"] = "MMF"


def janoschek(x, a, beta, k, delta):
    """
    http://www.pisces-conservation.com/growthhelp/janoschek.htm
    """
    return a - (a - beta) * np.exp(-k * x ** delta)


all_models["janoschek"] = janoschek
model_defaults["janoschek"] = {"a": 0.73, "beta": 0.07, "k": 0.355, "delta": 0.46}
display_name_mapping["janoschek"] = "Janoschek"


def ilog2(x, c, a):
    x = 1 + x
    assert np.all(x > 1)
    return c - a / np.log(x)


all_models["ilog2"] = ilog2
model_defaults["ilog2"] = {"a": 0.43, "c": 0.78}
display_name_mapping["ilog2"] = "ilog$_2$"


def dr_hill_zero_background(x, theta, eta, kappa):
    x_eta = x ** eta
    return (theta * x_eta) / (kappa ** eta + x_eta)


all_models["dr_hill_zero_background"] = dr_hill_zero_background
model_defaults["dr_hill_zero_background"] = {
    "theta": 0.772320,
    "eta": 0.586449,
    "kappa": 2.460843,
}
display_name_mapping["dr_hill_zero_background"] = "Hill$_3$"


def logx_linear(x, a, b):
    x = np.log(x)
    return a * x + b


all_models["logx_linear"] = logx_linear
model_defaults["logx_linear"] = {"a": 0.378106, "b": 0.046506}
display_name_mapping["logx_linear"] = "log x linear"


def vap(x, a, b, c):
    """Vapor pressure model"""
    return np.exp(a + b / x + c * np.log(x))


all_models["vap"] = vap
model_defaults["vap"] = {"a": -0.622028, "c": 0.042322, "b": -0.470050}
display_name_mapping["vap"] = "vapor pressure"


def loglog_linear(x, a, b):
    x = np.log(x)
    return np.log(a * x + b)


all_models["loglog_linear"] = loglog_linear
display_name_mapping["loglog_linear"] = "log log linear"


# Models that we chose not to use in the ensembles/model combinations:

# source: http://aclweb.org/anthology//P/P12/P12-1003.pdf
def exp3(x, c, a, b):
    return c - np.exp(-a * x + b)


all_models["exp3"] = exp3
model_defaults["exp3"] = {"c": 0.7, "a": 0.01, "b": -1}
display_name_mapping["exp3"] = "exp$_3$"


def exp4(x, c, a, b, alpha):
    return c - np.exp(-a * (x ** alpha) + b)


all_models["exp4"] = exp4
model_defaults["exp4"] = {"c": 0.7, "a": 0.8, "b": -0.8, "alpha": 0.3}
display_name_mapping["exp4"] = "exp$_4$"


# not bounded!
# def logy_linear(x, a, b):
#    return np.log(a*x + b)
# all_models["logy_linear"] = logy_linear


def pow2(x, a, alpha):
    return a * x ** (-alpha)


all_models["pow2"] = pow2
model_defaults["pow2"] = {"a": 0.1, "alpha": -0.3}
display_name_mapping["pow2"] = "pow$_2$"


def pow4(x, c, a, b, alpha):
    return c - (a * x + b) ** -alpha


all_models["pow4"] = pow4
model_defaults["pow4"] = {"alpha": 0.1, "a": 200, "b": 0.0, "c": 0.8}
display_name_mapping["pow4"] = "pow$_4$"


def sat_growth(x, a, b):
    return a * x / (b + x)


all_models["sat_growth"] = sat_growth
model_defaults["sat_growth"] = {"a": 0.7, "b": 20}
display_name_mapping["sat_growth"] = "saturated growth rate"


def dr_hill(x, alpha, theta, eta, kappa):
    return alpha + (theta * (x ** eta)) / (kappa ** eta + x ** eta)


all_models["dr_hill"] = dr_hill
model_defaults["dr_hill"] = {
    "alpha": 0.1,
    "theta": 0.772320,
    "eta": 0.586449,
    "kappa": 2.460843,
}
display_name_mapping["dr_hill"] = "Hill$_4$"


def gompertz(x, a, b, c):
    """
    Gompertz growth function.

    sigmoidal family
    a is the upper asymptote, since
    b, c are negative numbers
    b sets the displacement along the x axis (translates the graph to the left or right)
    c sets the growth rate (y scaling)

    e.g. used to model the growth of tumors

    http://en.wikipedia.org/wiki/Gompertz_function
    """
    return a * np.exp(-b * np.exp(-c * x))
    # return a + b * np.exp(np.exp(-k*(x-i)))


all_models["gompertz"] = gompertz
model_defaults["gompertz"] = {"a": 0.8, "b": 1000, "c": 0.05}
display_name_mapping["gompertz"] = "Gompertz"


def logistic_curve(x, a, k, b):
    """
    a: asymptote
    k:
    b: inflection point
    http://www.pisces-conservation.com/growthhelp/logistic_curve.htm
    """
    return a / (1.0 + np.exp(-k * (x - b)))


all_models["logistic_curve"] = logistic_curve
model_defaults["logistic_curve"] = {"a": 0.8, "k": 0.01, "b": 1.0}
display_name_mapping["logistic_curve"] = "logistic curve"


def bertalanffy(x, a, k):
    """
    a: asymptote
    k: growth rate
    http://www.pisces-conservation.com/growthhelp/von_bertalanffy.htm
    """
    return a * (1.0 - np.exp(-k * x))


all_models["bertalanffy"] = bertalanffy
model_defaults["bertalanffy"] = {"a": 0.8, "k": 0.01}
display_name_mapping["bertalanffy"] = "Bertalanffy"

curve_combination_models_old = [
    "vap",
    "ilog2",
    "weibull",
    "pow3",
    "pow4",
    "loglog_linear",
    "mmf",
    "janoschek",
    "dr_hill_zero_background",
    "log_power",
    "exp4",
]

curve_combination_models_original = [
    "weibull",
    "pow4",
    "mmf",
    "pow3",
    "loglog_linear",
    "janoschek",
    "dr_hill_zero_background",
    "log_power",
    "exp4",
]

# note: removing some of the models was found to improve performance
curve_combination_models = [
    "mmf",
    "loglog_linear",
    "dr_hill_zero_background",
    "log_power",
]

curve_ensemble_models = [
    "vap",
    "ilog2",
    "weibull",
    "pow3",
    "pow4",
    "loglog_linear",
    "mmf",
    "janoschek",
    "dr_hill_zero_background",
    "log_power",
    "exp4",
]
