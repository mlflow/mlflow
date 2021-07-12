# pylint: disable=wrong-import-position
"""
The ``mlflux`` module provides a high-level "fluent" API for starting and managing mlflux runs.
For example:

.. code:: python

    import mlflux

    mlflux.start_run()
    mlflux.log_param("my", "param")
    mlflux.log_metric("score", 100)
    mlflux.end_run()

You can also use the context manager syntax like this:

.. code:: python

    with mlflux.start_run() as run:
        mlflux.log_param("my", "param")
        mlflux.log_metric("score", 100)

which automatically terminates the run at the end of the ``with`` block.

The fluent tracking API is not currently threadsafe. Any concurrent callers to the tracking API must
implement mutual exclusion manually.

For a lower level API, see the :py:mod:`mlflux.tracking` module.
"""
from mlflux.version import VERSION as __version__  # pylint: disable=unused-import
from mlflux.utils.logging_utils import _configure_mlflow_loggers
import mlflux.tracking._model_registry.fluent
import mlflux.tracking.fluent

# Filter annoying Cython warnings that serve no good purpose, and so before
# importing other modules.
# See: https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d7
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # noqa: E402
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # noqa: E402

import mlflux.projects as projects  # noqa: E402
import mlflux.tracking as tracking  # noqa: E402

# model flavors
_model_flavors_supported = []
try:
    # pylint: disable=unused-import
    import mlflux.catboost as catboost  # noqa: E402
    import mlflux.fastai as fastai  # noqa: E402
    import mlflux.gluon as gluon  # noqa: E402
    import mlflux.h2o as h2o  # noqa: E402
    import mlflux.keras as keras  # noqa: E402
    import mlflux.lightgbm as lightgbm  # noqa: E402
    import mlflux.mleap as mleap  # noqa: E402
    import mlflux.onnx as onnx  # noqa: E402
    import mlflux.pyfunc as pyfunc  # noqa: E402
    import mlflux.pytorch as pytorch  # noqa: E402
    import mlflux.sklearn as sklearn  # noqa: E402
    import mlflux.spacy as spacy  # noqa: E402
    import mlflux.spark as spark  # noqa: E402
    import mlflux.statsmodels as statsmodels  # noqa: E402
    import mlflux.tensorflow as tensorflow  # noqa: E402
    import mlflux.xgboost as xgboost  # noqa: E402
    import mlflux.shap as shap  # noqa: E402
    import mlflux.pyspark as pyspark  # noqa: E402
    import mlflux.paddle as paddle

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
    ]
except ImportError as e:
    # We are conditional loading these commands since the skinny client does
    # not support them due to the pandas and numpy dependencies of mlflux Models
    pass


_configure_mlflow_loggers(root_module_name=__name__)

# TODO: Uncomment this block when deprecating Python 3.6 support
# _major = 3
# _minor = 6
# _deprecated_version = (_major, _minor)
# _min_supported_version = (_major, _minor + 1)

# if sys.version_info[:2] == _deprecated_version:
#     warnings.warn(
#         "mlflux support for Python {dep_ver} is deprecated and will be dropped in "
#         "an upcoming release. At that point, existing Python {dep_ver} workflows "
#         "that use mlflux will continue to work without modification, but Python {dep_ver} "
#         "users will no longer get access to the latest mlflux features and bugfixes. "
#         "We recommend that you upgrade to Python {min_ver} or newer.".format(
#             dep_ver=".".join(map(str, _deprecated_version)),
#             min_ver=".".join(map(str, _min_supported_version)),
#         ),
#         FutureWarning,
#         stacklevel=2,
#     )

ActiveRun = mlflux.tracking.fluent.ActiveRun
log_param = mlflux.tracking.fluent.log_param
log_metric = mlflux.tracking.fluent.log_metric
set_tag = mlflux.tracking.fluent.set_tag
delete_tag = mlflux.tracking.fluent.delete_tag
log_artifacts = mlflux.tracking.fluent.log_artifacts
log_artifact = mlflux.tracking.fluent.log_artifact
log_text = mlflux.tracking.fluent.log_text
log_dict = mlflux.tracking.fluent.log_dict
log_image = mlflux.tracking.fluent.log_image
log_figure = mlflux.tracking.fluent.log_figure
active_run = mlflux.tracking.fluent.active_run
get_run = mlflux.tracking.fluent.get_run
start_run = mlflux.tracking.fluent.start_run
end_run = mlflux.tracking.fluent.end_run
search_runs = mlflux.tracking.fluent.search_runs
list_run_infos = mlflux.tracking.fluent.list_run_infos
get_artifact_uri = mlflux.tracking.fluent.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
set_registry_uri = tracking.set_registry_uri
get_experiment = mlflux.tracking.fluent.get_experiment
get_experiment_by_name = mlflux.tracking.fluent.get_experiment_by_name
list_experiments = mlflux.tracking.fluent.list_experiments
get_tracking_uri = tracking.get_tracking_uri
get_registry_uri = tracking.get_registry_uri
create_experiment = mlflux.tracking.fluent.create_experiment
set_experiment = mlflux.tracking.fluent.set_experiment
log_params = mlflux.tracking.fluent.log_params
log_metrics = mlflux.tracking.fluent.log_metrics
set_tags = mlflux.tracking.fluent.set_tags
delete_experiment = mlflux.tracking.fluent.delete_experiment
delete_run = mlflux.tracking.fluent.delete_run
register_model = mlflux.tracking._model_registry.fluent.register_model
autolog = mlflux.tracking.fluent.autolog


run = projects.run

__all__ = [
    "ActiveRun",
    "log_param",
    "log_params",
    "log_metric",
    "log_metrics",
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
    "get_experiment",
    "get_experiment_by_name",
    "list_experiments",
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
] + _model_flavors_supported
