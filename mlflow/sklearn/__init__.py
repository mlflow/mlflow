"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for scikit-learn models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""
from mlflow.sklearn.sklearn import (
    _SklearnTrainingSession,
    _SklearnModelWrapper,
    get_default_pip_requirements,
    get_default_conda_env,
    save_model,
    log_model,
    load_model,
    autolog,
)

__all__ = [
    "_SklearnTrainingSession",
    "_SklearnModelWrapper",
    "get_default_pip_requirements",
    "get_default_conda_env",
    "save_model",
    "log_model",
    "load_model",
    "autolog",
]
