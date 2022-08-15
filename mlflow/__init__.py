# pylint: disable=wrong-import-position
"""
The ``mlflow`` module provides a high-level "fluent" API for starting and managing MLflow runs.
For example:

.. code:: python

    import mlflow

    mlflow.start_run()
    mlflow.log_param("my", "param")
    mlflow.log_metric("score", 100)
    mlflow.end_run()

You can also use the context manager syntax like this:

.. code:: python

    with mlflow.start_run() as run:
        mlflow.log_param("my", "param")
        mlflow.log_metric("score", 100)

which automatically terminates the run at the end of the ``with`` block.

The fluent tracking API is not currently threadsafe. Any concurrent callers to the tracking API must
implement mutual exclusion manually.

For a lower level API, see the :py:mod:`mlflow.client` module.
"""
from mlflow.version import VERSION as __version__  # pylint: disable=unused-import
from mlflow.utils.logging_utils import _configure_mlflow_loggers
import mlflow.tracking._model_registry.fluent
import mlflow.tracking.fluent

# Filter annoying Cython warnings that serve no good purpose, and so before
# importing other modules.
# See: https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d7
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from mlflow import projects
from mlflow import tracking
import mlflow.models
import mlflow.artifacts
import mlflow.pipelines
import mlflow.client

# model flavors
_model_flavors_supported = []
try:
    # pylint: disable=unused-import
    from mlflow import catboost
    from mlflow import fastai
    from mlflow import gluon
    from mlflow import h2o
    from mlflow import keras
    from mlflow import lightgbm
    from mlflow import mleap
    from mlflow import onnx
    from mlflow import pyfunc
    from mlflow import pytorch
    from mlflow import sklearn
    from mlflow import spacy
    from mlflow import spark
    from mlflow import statsmodels
    from mlflow import tensorflow
    from mlflow import xgboost
    from mlflow import shap
    from mlflow import pyspark
    from mlflow import paddle
    from mlflow import prophet
    from mlflow import pmdarima
    from mlflow import diviner

    _model_flavors_supported = [
        "catboost",
        "fastai",
        "gluon",
        "h2o",
        "keras",
        "lightgbm",
        "mleap",
        "onnx",
        "pyfunc",
        "pytorch",
        "sklearn",
        "spacy",
        "spark",
        "statsmodels",
        "tensorflow",
        "xgboost",
        "shap",
        "paddle",
        "prophet",
        "pmdarima",
        "diviner",
    ]
except ImportError as e:
    # We are conditional loading these commands since the skinny client does
    # not support them due to the pandas and numpy dependencies of MLflow Models
    pass


_configure_mlflow_loggers(root_module_name=__name__)

# TODO: Comment out this block when we deprecate support for python 3.7.
# _major = 3
# _minor = 7
# _deprecated_version = (_major, _minor)
# _min_supported_version = (_major, _minor + 1)

# if sys.version_info[:2] == _deprecated_version:
#     warnings.warn(
#         "MLflow support for Python {dep_ver} is deprecated and will be dropped in "
#         "an upcoming release. At that point, existing Python {dep_ver} workflows "
#         "that use MLflow will continue to work without modification, but Python {dep_ver} "
#         "users will no longer get access to the latest MLflow features and bugfixes. "
#         "We recommend that you upgrade to Python {min_ver} or newer.".format(
#             dep_ver=".".join(map(str, _deprecated_version)),
#             min_ver=".".join(map(str, _min_supported_version)),
#         ),
#         FutureWarning,
#         stacklevel=2,
#     )

ActiveRun = mlflow.tracking.fluent.ActiveRun
log_param = mlflow.tracking.fluent.log_param
log_metric = mlflow.tracking.fluent.log_metric
set_tag = mlflow.tracking.fluent.set_tag
delete_tag = mlflow.tracking.fluent.delete_tag
log_artifacts = mlflow.tracking.fluent.log_artifacts
log_artifact = mlflow.tracking.fluent.log_artifact
log_text = mlflow.tracking.fluent.log_text
log_dict = mlflow.tracking.fluent.log_dict
log_image = mlflow.tracking.fluent.log_image
log_figure = mlflow.tracking.fluent.log_figure
active_run = mlflow.tracking.fluent.active_run
get_run = mlflow.tracking.fluent.get_run
start_run = mlflow.tracking.fluent.start_run
end_run = mlflow.tracking.fluent.end_run
search_runs = mlflow.tracking.fluent.search_runs
list_run_infos = mlflow.tracking.fluent.list_run_infos
get_artifact_uri = mlflow.tracking.fluent.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
set_registry_uri = tracking.set_registry_uri
get_experiment = mlflow.tracking.fluent.get_experiment
get_experiment_by_name = mlflow.tracking.fluent.get_experiment_by_name
list_experiments = mlflow.tracking.fluent.list_experiments
search_experiments = mlflow.tracking.fluent.search_experiments
get_tracking_uri = tracking.get_tracking_uri
get_registry_uri = tracking.get_registry_uri
is_tracking_uri_set = tracking.is_tracking_uri_set
create_experiment = mlflow.tracking.fluent.create_experiment
set_experiment = mlflow.tracking.fluent.set_experiment
log_params = mlflow.tracking.fluent.log_params
log_metrics = mlflow.tracking.fluent.log_metrics
set_experiment_tags = mlflow.tracking.fluent.set_experiment_tags
set_experiment_tag = mlflow.tracking.fluent.set_experiment_tag
set_tags = mlflow.tracking.fluent.set_tags
delete_experiment = mlflow.tracking.fluent.delete_experiment
delete_run = mlflow.tracking.fluent.delete_run
register_model = mlflow.tracking._model_registry.fluent.register_model
autolog = mlflow.tracking.fluent.autolog
evaluate = mlflow.models.evaluate
last_active_run = mlflow.tracking.fluent.last_active_run
MlflowClient = mlflow.client.MlflowClient

run = projects.run

__all__ = [
    "ActiveRun",
    "log_param",
    "log_params",
    "log_metric",
    "log_metrics",
    "set_experiment_tags",
    "set_experiment_tag",
    "set_tag",
    "set_tags",
    "delete_tag",
    "log_artifacts",
    "log_artifact",
    "log_text",
    "log_dict",
    "log_figure",
    "log_image",
    "active_run",
    "start_run",
    "end_run",
    "search_runs",
    "get_artifact_uri",
    "get_tracking_uri",
    "set_tracking_uri",
    "is_tracking_uri_set",
    "get_experiment",
    "get_experiment_by_name",
    "list_experiments",
    "search_experiments",
    "create_experiment",
    "set_experiment",
    "delete_experiment",
    "get_run",
    "delete_run",
    "run",
    "register_model",
    "get_registry_uri",
    "set_registry_uri",
    "list_run_infos",
    "autolog",
    "evaluate",
    "last_active_run",
    "MlflowClient",
] + _model_flavors_supported
