"""
The ``mlflow.models`` module provides an API for saving machine learning models in
"flavors" that can be understood by different downstream tools.

The built-in flavors are:

- :py:mod:`mlflow.pyfunc`
- :py:mod:`mlflow.h2o`
- :py:mod:`mlflow.keras`
- :py:mod:`mlflow.pytorch`
- :py:mod:`mlflow.sklearn`
- :py:mod:`mlflow.spark`
- :py:mod:`mlflow.tensorflow`

For details, see `MLflow Models <../models.html>`_.
"""

from .model import Model
from .flavor_backend import FlavorBackend
