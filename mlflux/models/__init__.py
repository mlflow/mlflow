"""
The ``mlflux.models`` module provides an API for saving machine learning models in
"flavors" that can be understood by different downstream tools.

The built-in flavors are:

- :py:mod:`mlflux.pyfunc`
- :py:mod:`mlflux.h2o`
- :py:mod:`mlflux.keras`
- :py:mod:`mlflux.lightgbm`
- :py:mod:`mlflux.pytorch`
- :py:mod:`mlflux.sklearn`
- :py:mod:`mlflux.spark`
- :py:mod:`mlflux.statsmodels`
- :py:mod:`mlflux.tensorflow`
- :py:mod:`mlflux.xgboost`
- :py:mod:`mlflux.spacy`
- :py:mod:`mlflux.fastai`
- :py:mod:`mlflux.paddle`

For details, see `mlflux Models <../models.html>`_.
"""

from .model import Model
from .flavor_backend import FlavorBackend
from .signature import ModelSignature, infer_signature
from .utils import ModelInputExample

__all__ = ["Model", "ModelSignature", "ModelInputExample", "infer_signature", "FlavorBackend"]
