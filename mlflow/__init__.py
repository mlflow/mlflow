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

from mlflow.version import VERSION

__version__ = VERSION
from mlflow import (
    artifacts,  # noqa: F401
    client,  # noqa: F401
    config,  # noqa: F401
    data,  # noqa: F401
    exceptions,  # noqa: F401
    models,  # noqa: F401
    projects,  # noqa: F401
    tracking,  # noqa: F401
)
from mlflow.environment_variables import MLFLOW_CONFIGURE_LOGGING
from mlflow.utils.lazy_load import LazyLoader
from mlflow.utils.logging_utils import _configure_mlflow_loggers

# Lazily load mlflow flavors to avoid excessive dependencies.
catboost = LazyLoader("mlflow.catboost", globals(), "mlflow.catboost")
diviner = LazyLoader("mlflow.diviner", globals(), "mlflow.diviner")
fastai = LazyLoader("mlflow.fastai", globals(), "mlflow.fastai")
gluon = LazyLoader("mlflow.gluon", globals(), "mlflow.gluon")
h2o = LazyLoader("mlflow.h2o", globals(), "mlflow.h2o")
johnsnowlabs = LazyLoader("mlflow.johnsnowlabs", globals(), "mlflow.johnsnowlabs")
keras = LazyLoader("mlflow.keras", globals(), "mlflow.keras")
langchain = LazyLoader("mlflow.langchain", globals(), "mlflow.langchain")
lightgbm = LazyLoader("mlflow.lightgbm", globals(), "mlflow.lightgbm")
llm = LazyLoader("mlflow.llm", globals(), "mlflow.llm")
metrics = LazyLoader("mlflow.metrics", globals(), "mlflow.metrics")
mleap = LazyLoader("mlflow.mleap", globals(), "mlflow.mleap")
onnx = LazyLoader("mlflow.onnx", globals(), "mlflow.onnx")
openai = LazyLoader("mlflow.openai", globals(), "mlflow.openai")
paddle = LazyLoader("mlflow.paddle", globals(), "mlflow.paddle")
pmdarima = LazyLoader("mlflow.pmdarima", globals(), "mlflow.pmdarima")
promptflow = LazyLoader("mlflow.promptflow", globals(), "mlflow.promptflow")
prophet = LazyLoader("mlflow.prophet", globals(), "mlflow.prophet")
promptlab = LazyLoader("mlflow.promptlab", globals(), "mlflow.promptlab")
pyfunc = LazyLoader("mlflow.pyfunc", globals(), "mlflow.pyfunc")
pyspark = LazyLoader("mlflow.pyspark", globals(), "mlflow.pyspark")
pytorch = LazyLoader("mlflow.pytorch", globals(), "mlflow.pytorch")
rfunc = LazyLoader("mlflow.rfunc", globals(), "mlflow.rfunc")
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

if MLFLOW_CONFIGURE_LOGGING.get() is True:
    _configure_mlflow_loggers(root_module_name=__name__)

from mlflow.client import MlflowClient

# For backward compatibility, we expose the following functions and classes at the top level in
# addition to `mlflow.config`.
from mlflow.config import (
    disable_system_metrics_logging,
    enable_system_metrics_logging,
    get_registry_uri,
    get_tracking_uri,
    is_tracking_uri_set,
    set_registry_uri,
    set_system_metrics_node_id,
    set_system_metrics_samples_before_logging,
    set_system_metrics_sampling_interval,
    set_tracking_uri,
)
from mlflow.exceptions import MlflowException
from mlflow.models import evaluate
from mlflow.projects import run
from mlflow.tracing.fluent import (
    get_current_active_span,
    get_last_active_trace,
    get_trace,
    search_traces,
    start_span,
    trace,
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
    flush_artifact_async_logging,
    flush_async_logging,
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
from mlflow.tracking.multimedia import Image
from mlflow.utils.async_logging.run_operations import RunOperations  # noqa: F401
from mlflow.utils.credentials import login
from mlflow.utils.doctor import doctor

__all__ = [
    "ActiveRun",
    "MlflowClient",
    "MlflowException",
    "active_run",
    "autolog",
    "create_experiment",
    "delete_experiment",
    "delete_run",
    "delete_tag",
    "disable_system_metrics_logging",
    "doctor",
    "enable_system_metrics_logging",
    "end_run",
    "evaluate",
    "flush_async_logging",
    "flush_artifact_async_logging",
    "get_artifact_uri",
    "get_experiment",
    "get_experiment_by_name",
    "get_last_active_trace",
    "get_parent_run",
    "get_registry_uri",
    "get_run",
    "get_tracking_uri",
    "is_tracking_uri_set",
    "last_active_run",
    "load_table",
    "log_artifact",
    "log_artifacts",
    "log_dict",
    "log_figure",
    "log_image",
    "log_input",
    "log_metric",
    "log_metrics",
    "log_param",
    "log_params",
    "log_table",
    "log_text",
    "login",
    "pyfunc",
    "register_model",
    "run",
    "search_experiments",
    "search_model_versions",
    "search_registered_models",
    "search_runs",
    "set_experiment",
    "set_experiment_tag",
    "set_experiment_tags",
    "set_registry_uri",
    "set_system_metrics_node_id",
    "set_system_metrics_samples_before_logging",
    "set_system_metrics_sampling_interval",
    "set_tag",
    "set_tags",
    "set_tracking_uri",
    "start_run",
    "Image",
    # Tracing Fluent APIs
    "get_current_active_span",
    "get_trace",
    "search_traces",
    "start_span",
    "trace",
]


# `mlflow.gateway` depends on optional dependencies such as pydantic, psutil, and has version
# restrictions for dependencies. Importing this module fails if they are not installed or
# if invalid versions of these required packages are installed.
with contextlib.suppress(Exception):
    from mlflow import gateway  # noqa: F401

    __all__.append("gateway")
