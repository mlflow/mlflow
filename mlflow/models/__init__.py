"""
The ``mlflow.models`` module provides an API for saving machine learning models in
"flavors" that can be understood by different downstream tools.

The built-in flavors are:

- :py:mod:`mlflow.pyfunc`
- :py:mod:`mlflow.h2o`
- :py:mod:`mlflow.lightgbm`
- :py:mod:`mlflow.pytorch`
- :py:mod:`mlflow.sklearn`
- :py:mod:`mlflow.spark`
- :py:mod:`mlflow.statsmodels`
- :py:mod:`mlflow.tensorflow`
- :py:mod:`mlflow.xgboost`
- :py:mod:`mlflow.spacy`
- :py:mod:`mlflow.fastai`
- :py:mod:`mlflow.paddle`

For details, see `MLflow Models <../models.html>`_.
"""

from .model import Model, get_model_info
from .flavor_backend import FlavorBackend
from ..utils.environment import infer_pip_requirements
from .evaluation import (
    evaluate,
    make_metric,
    EvaluationMetric,
    EvaluationArtifact,
    EvaluationResult,
    list_evaluators,
    MetricThreshold,
)

from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager


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
    from .signature import ModelSignature, infer_signature  # pylint: disable=unused-import
    from .utils import ModelInputExample, validate_schema  # pylint: disable=unused-import
    from .utils import add_libraries_to_model  # pylint: disable=unused-import

    __all__ += [
        "ModelSignature",
        "ModelInputExample",
        "infer_signature",
        "validate_schema",
        "add_libraries_to_model",
    ]
except ImportError:
    pass
