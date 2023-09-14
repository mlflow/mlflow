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
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import Model, get_model_info
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.environment import infer_pip_requirements


def build_docker(
    model_uri=None,
    name="mlflow-pyfunc",
    env_manager=_EnvManager.VIRTUALENV,
    mlflow_home=None,
    install_mlflow=False,
    enable_mlserver=False,
):
    """
    Builds a Docker image whose default entrypoint serves an MLflow model at port 8080, using the
    python_function flavor. The container serves the model referenced by ``model_uri``, if
    specified. If ``model_uri`` is not specified, an MLflow Model directory must be mounted as a
    volume into the /opt/ml/model directory in the container.

    .. warning::

        If ``model_uri`` is unspecified, the resulting image doesn't support serving models with
        the RFunc or Java MLeap model servers.

    NB: by default, the container will start nginx and gunicorn processes. If you don't need the
    nginx process to be started (for instance if you deploy your container to Google Cloud Run),
    you can disable it via the DISABLE_NGINX environment variable:

    .. code:: bash

        docker run -p 5001:8080 -e DISABLE_NGINX=true "my-image-name"

    See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information on the
    'python_function' flavor.
    """
    get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager).build_image(
        model_uri,
        name,
        mlflow_home=mlflow_home,
        install_mlflow=install_mlflow,
        enable_mlserver=enable_mlserver,
    )


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
    "list_evaluators",
    "MetricThreshold",
    "build_docker",
]


# Under skinny-mlflow requirements, the following packages cannot be imported
# because of lack of numpy/pandas library, so wrap them with try...except block
try:
    from mlflow.models.signature import ModelSignature, infer_signature, set_signature
    from mlflow.models.utils import ModelInputExample, add_libraries_to_model, validate_schema

    __all__ += [
        "ModelSignature",
        "ModelInputExample",
        "infer_signature",
        "validate_schema",
        "add_libraries_to_model",
        "set_signature",
    ]
except ImportError:
    pass
