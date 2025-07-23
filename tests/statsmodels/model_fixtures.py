from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import toeplitz
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


class ModelWithResults(NamedTuple):
    model: Any
    alg: Any
    inference_dataframe: Any


"""
    Fixtures for a number of models available in statsmodels
    https://www.statsmodels.org/dev/api.html
"""


def ols_model(**kwargs):
    # Ordinary Least Squares (OLS)
    np.random.seed(9876789)
    nsamples = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x**2))
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=nsamples)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e

    ols = sm.OLS(y, X)
    model = ols.fit(**kwargs)

    return ModelWithResults(model=model, alg=ols, inference_dataframe=X)


def ols_model_signature():
    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float64"), (-1, 3))]),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1,))]),
    )


def failing_logit_model():
    X = pd.DataFrame(
        {
            "x0": np.array([2.0, 3.0, 1.0, 2.0, 20.0, 30.0, 10.0, 20.0]),
            "x1": np.array([2.0, 3.0, 1.0, 2.0, 20.0, 30.0, 10.0, 20.0]),
        },
        columns=["x0", "x1"],
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # building the model and fitting the data
    log_reg = sm.Logit(y, X)
    model = log_reg.fit()

    return ModelWithResults(model=model, alg=log_reg, inference_dataframe=X)


def get_dataset(name):
    dataset_module = getattr(sm.datasets, name)
    data = dataset_module.load()
    data.exog = np.asarray(data.exog)
    data.endog = np.asarray(data.endog)
    return data


def gls_model():
    # Generalized Least Squares (GLS)
    data = get_dataset("longley")
    data.exog = sm.add_constant(data.exog)
    ols_resid = sm.OLS(data.endog, data.exog).fit().resid
    res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
    rho = res_fit.params
    order = toeplitz(np.arange(16))
    sigma = rho**order
    gls = sm.GLS(data.endog, data.exog, sigma=sigma)
    model = gls.fit()

    return ModelWithResults(model=model, alg=gls, inference_dataframe=data.exog)


def glsar_model():
    # Generalized Least Squares with AR covariance structure
    X = range(1, 8)
    X = sm.add_constant(X)
    Y = [1, 3, 4, 5, 8, 10, 9]
    glsar = sm.GLSAR(Y, X, rho=2)
    model = glsar.fit()

    return ModelWithResults(model=model, alg=glsar, inference_dataframe=X)


def wls_model():
    # Weighted Least Squares
    Y = [1, 3, 4, 5, 2, 3, 4]
    X = range(1, 8)
    X = sm.add_constant(X)
    wls = sm.WLS(Y, X, weights=list(range(1, 8)))
    model = wls.fit()

    return ModelWithResults(model=model, alg=wls, inference_dataframe=X)


def recursivels_model():
    # Recursive Least Squares
    dta = sm.datasets.copper.load_pandas().data
    dta.index = pd.date_range("1951-01-01", "1975-01-01", freq="AS")
    endog = dta.WORLDCONSUMPTION

    # To the regressors in the dataset, we add a column of ones for an intercept
    exog = sm.add_constant(dta[["COPPERPRICE", "INCOMEINDEX", "ALUMPRICE", "INVENTORYINDEX"]])
    rls = sm.RecursiveLS(endog, exog)
    model = rls.fit()

    inference_dataframe = pd.DataFrame([["1951-01-01", "1975-01-01"]], columns=["start", "end"])
    return ModelWithResults(model=model, alg=rls, inference_dataframe=inference_dataframe)


def rolling_ols_model():
    # Rolling Ordinary Least Squares (Rolling OLS)
    from statsmodels.regression.rolling import RollingOLS

    data = get_dataset("longley")
    exog = sm.add_constant(data.exog, prepend=False)
    rolling_ols = RollingOLS(data.endog, exog)
    model = rolling_ols.fit(reset=50)

    return ModelWithResults(model=model, alg=rolling_ols, inference_dataframe=exog)


def rolling_wls_model():
    # Rolling Weighted Least Squares (Rolling WLS)
    from statsmodels.regression.rolling import RollingWLS

    data = get_dataset("longley")
    exog = sm.add_constant(data.exog, prepend=False)
    rolling_wls = RollingWLS(data.endog, exog)
    model = rolling_wls.fit(reset=50)

    return ModelWithResults(model=model, alg=rolling_wls, inference_dataframe=exog)


def gee_model():
    # Example taken from
    # https://www.statsmodels.org/devel/examples/notebooks/generated/gee_nested_simulation.html
    np.random.seed(9876789)
    p = 5
    groups_var = 1
    level1_var = 2
    level2_var = 3
    resid_var = 4
    n_groups = 100
    group_size = 20
    level1_size = 10
    level2_size = 5
    n = n_groups * group_size * level1_size * level2_size
    xmat = np.random.normal(size=(n, p))

    # Construct labels showing which group each observation belongs to at each level.
    groups_ix = np.kron(np.arange(n // group_size), np.ones(group_size)).astype(int)
    level1_ix = np.kron(np.arange(n // level1_size), np.ones(level1_size)).astype(int)
    level2_ix = np.kron(np.arange(n // level2_size), np.ones(level2_size)).astype(int)

    # Simulate the random effects.
    groups_re = np.sqrt(groups_var) * np.random.normal(size=n // group_size)
    level1_re = np.sqrt(level1_var) * np.random.normal(size=n // level1_size)
    level2_re = np.sqrt(level2_var) * np.random.normal(size=n // level2_size)

    # Simulate the response variable
    y = groups_re[groups_ix] + level1_re[level1_ix] + level2_re[level2_ix]
    y += np.sqrt(resid_var) * np.random.normal(size=n)

    # Put everything into a dataframe.
    df = pd.DataFrame(xmat, columns=[f"x{j}" for j in range(p)])
    df["y"] = y + xmat[:, 0] - xmat[:, 3]
    df["groups_ix"] = groups_ix
    df["level1_ix"] = level1_ix
    df["level2_ix"] = level2_ix

    # Fit the model
    cs = sm.cov_struct.Nested()
    dep_fml = "0 + level1_ix + level2_ix"
    gee = sm.GEE.from_formula(
        "y ~ x0 + x1 + x2 + x3 + x4", cov_struct=cs, dep_data=dep_fml, groups="groups_ix", data=df
    )
    model = gee.fit()

    return ModelWithResults(model=model, alg=gee, inference_dataframe=df)


def glm_model():
    # Generalized Linear Model (GLM)
    data = get_dataset("scotland")
    data.exog = sm.add_constant(data.exog)
    glm = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
    model = glm.fit()

    return ModelWithResults(model=model, alg=glm, inference_dataframe=data.exog)


def glmgam_model():
    # Generalized Additive Model (GAM)
    from statsmodels.gam.tests.test_penalized import df_autos

    x_spline = df_autos[["weight", "hp"]]
    bs = sm.gam.BSplines(x_spline, df=[12, 10], degree=[3, 3])
    alpha = np.array([21833888.8, 6460.38479])
    gam_bs = sm.GLMGam.from_formula(
        "city_mpg ~ fuel + drive", data=df_autos, smoother=bs, alpha=alpha
    )
    model = gam_bs.fit()

    return ModelWithResults(model=model, alg=gam_bs, inference_dataframe=df_autos)


def arma_model():
    # Autoregressive Moving Average (ARMA)
    np.random.seed(12345)
    arparams = np.array([1, -0.75, 0.25])
    maparams = np.array([1, 0.65, 0.35])
    nobs = 250
    y = arma_generate_sample(arparams, maparams, nobs)
    dates = pd.date_range("1980-1-1", freq="M", periods=nobs)
    y = pd.Series(y, index=dates)

    arima = ARIMA(y, order=(2, 0, 2), trend="n")
    model = arima.fit()
    inference_dataframe = pd.DataFrame([["1999-06-30", "2001-05-31"]], columns=["start", "end"])

    return ModelWithResults(model=model, alg=arima, inference_dataframe=inference_dataframe)
