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
import contextlib

# Filter annoying Cython warnings that serve no good purpose, and so before
# importing other modules.
# See: https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d7
import warnings

from mlflow.utils.lazy_load import LazyLoader
from mlflow.utils.logging_utils import _configure_mlflow_loggers
from mlflow.version import VERSION as __version__  # noqa: F401

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from mlflow import (
    artifacts,  # noqa: F401
    client,  # noqa: F401
    data,  # noqa: F401
    exceptions,  # noqa: F401
    models,  # noqa: F401
    projects,  # noqa: F401
    tracking,  # noqa: F401
)

# Lazily load mlflow flavors to avoid excessive dependencies.
catboost = LazyLoader("mlflow.catboost", globals(), "mlflow.catboost")
diviner = LazyLoader("mlflow.diviner", globals(), "mlflow.diviner")
fastai = LazyLoader("mlflow.fastai", globals(), "mlflow.fastai")
gluon = LazyLoader("mlflow.gluon", globals(), "mlflow.gluon")
h2o = LazyLoader("mlflow.h2o", globals(), "mlflow.h2o")
johnsnowlabs = LazyLoader("mlflow.johnsnowlabs", globals(), "mlflow.johnsnowlabs")
langchain = LazyLoader("mlflow.langchain", globals(), "mlflow.langchain")
lightgbm = LazyLoader("mlflow.lightgbm", globals(), "mlflow.lightgbm")
llm = LazyLoader("mlflow.llm", globals(), "mlflow.llm")
mleap = LazyLoader("mlflow.mleap", globals(), "mlflow.mleap")
onnx = LazyLoader("mlflow.onnx", globals(), "mlflow.onnx")
openai = LazyLoader("mlflow.openai", globals(), "mlflow.openai")
paddle = LazyLoader("mlflow.paddle", globals(), "mlflow.paddle")
pmdarima = LazyLoader("mlflow.pmdarima", globals(), "mlflow.pmdarima")
prophet = LazyLoader("mlflow.prophet", globals(), "mlflow.prophet")
pyfunc = LazyLoader("mlflow.pyfunc", globals(), "mlflow.pyfunc")
pyspark = LazyLoader("mlflow.pyspark", globals(), "mlflow.pyspark")
pytorch = LazyLoader("mlflow.pytorch", globals(), "mlflow.pytorch")
recipes = LazyLoader("mlflow.recipes", globals(), "mlflow.recipes")
sentence_transformers = LazyLoader(
    "mlflow.sentence_transformers",
    globals(),
    "mlflow.sentence_transformers",
)
shap = LazyLoader("mlflow.shap", globals(), "mlflow.shap")
sklearn = LazyLoader("mlflow.sklearn", globals(), "mlflow.sklearn")
spacy = LazyLoader("mlflow.spacy", globals(), "mlflow.spacy")
spark = LazyLoader("mlflow.spark", globals(), "mlflow.spark")
statsmodels = LazyLoader("mlflow.statsmodels", globals(), "mlflow.statsmodels")
tensorflow = LazyLoader("mlflow.tensorflow", globals(), "mlflow.tensorflow")
transformers = LazyLoader("mlflow.transformers", globals(), "mlflow.transformers")
xgboost = LazyLoader("mlflow.xgboost", globals(), "mlflow.xgboost")

_configure_mlflow_loggers(root_module_name=__name__)

# TODO: Comment out this block when we deprecate support for python 3.8.
# _major = 3
# _minor = 8
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

from mlflow._doctor import doctor
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import evaluate
from mlflow.projects import run
from mlflow.tracking import (
    get_registry_uri,
    get_tracking_uri,
    is_tracking_uri_set,
    set_registry_uri,
    set_tracking_uri,
)
from mlflow.tracking._model_registry.fluent import (
    register_model,
    search_model_versions,
    search_registered_models,
)
from mlflow.tracking.fluent import (
    ActiveRun,
    active_run,
    autolog,
    create_experiment,
    delete_experiment,
    delete_run,
    delete_tag,
    end_run,
    get_artifact_uri,
    get_experiment,
    get_experiment_by_name,
    get_parent_run,
    get_run,
    last_active_run,
    load_table,
    log_artifact,
    log_artifacts,
    log_dict,
    log_figure,
    log_image,
    log_input,
    log_metric,
    log_metrics,
    log_param,
    log_params,
    log_table,
    log_text,
    search_experiments,
    search_runs,
    set_experiment,
    set_experiment_tag,
    set_experiment_tags,
    set_tag,
    set_tags,
    start_run,
)

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
    "log_table",
    "load_table",
    "log_image",
    "log_input",
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
    "search_experiments",
    "search_registered_models",
    "search_model_versions",
    "create_experiment",
    "set_experiment",
    "delete_experiment",
    "get_run",
    "get_parent_run",
    "delete_run",
    "run",
    "register_model",
    "get_registry_uri",
    "set_registry_uri",
    "autolog",
    "evaluate",
    "last_active_run",
    "doctor",
    "MlflowClient",
    "MlflowException",
]


# `mlflow.gateway` depends on optional dependencies such as pydantic and has version
# restrictions for dependencies. Importing this module fails if they are not installed or
# if invalid versions of these required packages are installed.
with contextlib.suppress(Exception):
    from mlflow import gateway  # noqa: F401

    __all__.append("gateway")
