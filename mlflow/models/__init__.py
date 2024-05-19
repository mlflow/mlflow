"""
The ``mlflow.models`` module provides an API for saving machine learning models in
"flavors" that can be understood by different downstream tools.

The built-in flavors are:

- :py:mod:`mlflow.catboost`
- :py:mod:`mlflow.diviner`
- :py:mod:`mlflow.fastai`
- :py:mod:`mlflow.gluon`
- :py:mod:`mlflow.h2o`
- :py:mod:`mlflow.langchain`
- :py:mod:`mlflow.lightgbm`
- :py:mod:`mlflow.mleap`
- :py:mod:`mlflow.onnx`
- :py:mod:`mlflow.openai`
- :py:mod:`mlflow.paddle`
- :py:mod:`mlflow.pmdarima`
- :py:mod:`mlflow.prophet`
- :py:mod:`mlflow.pyfunc`
- :py:mod:`mlflow.pyspark.ml`
- :py:mod:`mlflow.pytorch`
- :py:mod:`mlflow.sklearn`
- :py:mod:`mlflow.spacy`
- :py:mod:`mlflow.spark`
- :py:mod:`mlflow.statsmodels`
- :py:mod:`mlflow.tensorflow`
- :py:mod:`mlflow.transformers`
- :py:mod:`mlflow.xgboost`

For details, see `MLflow Models <../models.html>`_.
"""
from mlflow.models.dependencies_schema import set_retriever_schema
from mlflow.models.evaluation import (
    EvaluationArtifact,
    EvaluationMetric,
    EvaluationResult,
    MetricThreshold,
    evaluate,
    list_evaluators,
    make_metric,
)
from mlflow.models.flavor_backend import FlavorBackend
from mlflow.models.model import Model, get_model_info, set_model
from mlflow.models.model_config import ModelConfig
from mlflow.models.python_api import build_docker
from mlflow.models.resources import Resource, ResourceType
from mlflow.utils.environment import infer_pip_requirements

__all__ = [
    "Model",
    "FlavorBackend",
    "infer_pip_requirements",
    "evaluate",
    "make_metric",
    "EvaluationMetric",
    "EvaluationArtifact",
    "EvaluationResult",
    "get_model_info",
    "set_model",
    "set_retriever_schema",
    "list_evaluators",
    "MetricThreshold",
    "build_docker",
    "Resource",
    "ResourceType",
    "ModelConfig",
]


# Under skinny-mlflow requirements, the following packages cannot be imported
# because of lack of numpy/pandas library, so wrap them with try...except block
try:
    from mlflow.models.python_api import predict
    from mlflow.models.signature import ModelSignature, infer_signature, set_signature
    from mlflow.models.utils import ModelInputExample, add_libraries_to_model, validate_schema

    __all__ += [
        "ModelSignature",
        "ModelInputExample",
        "infer_signature",
        "validate_schema",
        "add_libraries_to_model",
        "set_signature",
        "predict",
    ]
except ImportError:
    pass
