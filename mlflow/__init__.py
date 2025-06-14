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
from typing import TYPE_CHECKING

from mlflow.version import IS_TRACING_SDK_ONLY, VERSION

__version__ = VERSION

import mlflow.mismatch

# `check_version_mismatch` must be called here before importing any other modules
with contextlib.suppress(Exception):
    mlflow.mismatch._check_version_mismatch()

if not IS_TRACING_SDK_ONLY:
    from mlflow import (
        artifacts,  # noqa: F401
        client,  # noqa: F401
        config,  # noqa: F401
        data,  # noqa: F401
        exceptions,  # noqa: F401
        genai,  # noqa: F401
        models,  # noqa: F401
        projects,  # noqa: F401
        tracking,  # noqa: F401
    )

from mlflow import tracing  # noqa: F401
from mlflow.environment_variables import MLFLOW_CONFIGURE_LOGGING
from mlflow.exceptions import MlflowException
from mlflow.utils.lazy_load import LazyLoader
from mlflow.utils.logging_utils import _configure_mlflow_loggers

# Lazily load mlflow flavors to avoid excessive dependencies.
anthropic = LazyLoader("mlflow.anthropic", globals(), "mlflow.anthropic")
ag2 = LazyLoader("mlflow.ag2", globals(), "mlflow.ag2")
autogen = LazyLoader("mlflow.autogen", globals(), "mlflow.autogen")
bedrock = LazyLoader("mlflow.bedrock", globals(), "mlflow.bedrock")
catboost = LazyLoader("mlflow.catboost", globals(), "mlflow.catboost")
crewai = LazyLoader("mlflow.crewai", globals(), "mlflow.crewai")
diviner = LazyLoader("mlflow.diviner", globals(), "mlflow.diviner")
dspy = LazyLoader("mlflow.dspy", globals(), "mlflow.dspy")
gemini = LazyLoader("mlflow.gemini", globals(), "mlflow.gemini")
groq = LazyLoader("mlflow.groq", globals(), "mlflow.groq")
h2o = LazyLoader("mlflow.h2o", globals(), "mlflow.h2o")
johnsnowlabs = LazyLoader("mlflow.johnsnowlabs", globals(), "mlflow.johnsnowlabs")
keras = LazyLoader("mlflow.keras", globals(), "mlflow.keras")
langchain = LazyLoader("mlflow.langchain", globals(), "mlflow.langchain")
lightgbm = LazyLoader("mlflow.lightgbm", globals(), "mlflow.lightgbm")
litellm = LazyLoader("mlflow.litellm", globals(), "mlflow.litellm")
llama_index = LazyLoader("mlflow.llama_index", globals(), "mlflow.llama_index")
llm = LazyLoader("mlflow.llm", globals(), "mlflow.llm")
metrics = LazyLoader("mlflow.metrics", globals(), "mlflow.metrics")
mistral = LazyLoader("mlflow.mistral", globals(), "mlflow.mistral")
onnx = LazyLoader("mlflow.onnx", globals(), "mlflow.onnx")
openai = LazyLoader("mlflow.openai", globals(), "mlflow.openai")
paddle = LazyLoader("mlflow.paddle", globals(), "mlflow.paddle")
pmdarima = LazyLoader("mlflow.pmdarima", globals(), "mlflow.pmdarima")
promptflow = LazyLoader("mlflow.promptflow", globals(), "mlflow.promptflow")
prophet = LazyLoader("mlflow.prophet", globals(), "mlflow.prophet")
pydantic_ai = LazyLoader("mlflow.pydantic_ai", globals(), "mlflow.pydantic_ai")
pyfunc = LazyLoader("mlflow.pyfunc", globals(), "mlflow.pyfunc")
pyspark = LazyLoader("mlflow.pyspark", globals(), "mlflow.pyspark")
pytorch = LazyLoader("mlflow.pytorch", globals(), "mlflow.pytorch")
rfunc = LazyLoader("mlflow.rfunc", globals(), "mlflow.rfunc")
sentence_transformers = LazyLoader(
    "mlflow.sentence_transformers",
    globals(),
    "mlflow.sentence_transformers",
)
shap = LazyLoader("mlflow.shap", globals(), "mlflow.shap")
sklearn = LazyLoader("mlflow.sklearn", globals(), "mlflow.sklearn")
smolagents = LazyLoader("mlflow.smolagents", globals(), "mlflow.smolagents")
spacy = LazyLoader("mlflow.spacy", globals(), "mlflow.spacy")
spark = LazyLoader("mlflow.spark", globals(), "mlflow.spark")
statsmodels = LazyLoader("mlflow.statsmodels", globals(), "mlflow.statsmodels")
tensorflow = LazyLoader("mlflow.tensorflow", globals(), "mlflow.tensorflow")
# TxtAI integration is defined at https://github.com/neuml/mlflow-txtai
txtai = LazyLoader("mlflow.txtai", globals(), "mlflow_txtai")
transformers = LazyLoader("mlflow.transformers", globals(), "mlflow.transformers")
xgboost = LazyLoader("mlflow.xgboost", globals(), "mlflow.xgboost")

if TYPE_CHECKING:
    # Do not move this block above the lazy-loaded modules above.
    # All the lazy-loaded modules above must be imported here for code completion to work in IDEs.
    from mlflow import (  # noqa: F401
        ag2,
        anthropic,
        autogen,
        bedrock,
        catboost,
        crewai,
        diviner,
        dspy,
        gemini,
        groq,
        h2o,
        johnsnowlabs,
        keras,
        langchain,
        lightgbm,
        litellm,
        llama_index,
        llm,
        metrics,
        mistral,
        onnx,
        openai,
        paddle,
        pmdarima,
        promptflow,
        prophet,
        pydantic_ai,
        pyfunc,
        pyspark,
        pytorch,
        rfunc,
        sentence_transformers,
        shap,
        sklearn,
        smolagents,
        spacy,
        spark,
        statsmodels,
        tensorflow,
        transformers,
        xgboost,
    )

if MLFLOW_CONFIGURE_LOGGING.get() is True:
    _configure_mlflow_loggers(root_module_name=__name__)

# Core modules required for mlflow-tracing
from mlflow.tracing.assessment import (
    delete_assessment,
    get_assessment,
    log_assessment,
    log_expectation,
    log_feedback,
    override_feedback,
    update_assessment,
)
from mlflow.tracing.fluent import (
    add_trace,
    delete_trace_tag,
    get_active_trace_id,
    get_current_active_span,
    get_last_active_trace_id,
    get_trace,
    log_trace,
    search_traces,
    set_trace_tag,
    start_span,
    start_span_no_context,
    trace,
    update_current_trace,
)
from mlflow.tracking import (
    get_tracking_uri,
    is_tracking_uri_set,
    set_tracking_uri,
)
from mlflow.tracking.fluent import active_run, flush_trace_async_logging, set_experiment

# These are minimal set of APIs to be exposed via `mlflow-tracing` package.
# APIs listed here must not depend on dependencies that are not part of `mlflow-tracing` package.
__all__ = [
    "MlflowException",
    # Minimal tracking APIs required for tracing core functionality
    "set_experiment",
    "set_tracking_uri",
    "get_tracking_uri",
    "is_tracking_uri_set",
    # NB: Tracing SDK doesn't support using Runs, however, active_run is used heavily within
    # the autologging code base.
    "active_run",
    # Tracing APIs
    "add_trace",
    "delete_trace_tag",
    "flush_trace_async_logging",
    "get_active_trace_id",
    "get_current_active_span",
    "get_last_active_trace_id",
    "get_trace",
    "log_trace",
    "search_traces",
    "set_trace_tag",
    "start_span",
    "start_span_no_context",
    "trace",
    "update_current_trace",
    # Assessment APIs
    "get_assessment",
    "delete_assessment",
    "log_assessment",
    "update_assessment",
    "log_expectation",
    "log_feedback",
    "override_feedback",
]

# Only import these modules when mlflow or mlflow-skinny is installed i.e. not importing them
# when only mlflow-tracing is installed.
if not IS_TRACING_SDK_ONLY:
    from mlflow.client import MlflowClient

    # For backward compatibility, we expose the following functions and classes at the top level in
    # addition to `mlflow.config`.
    from mlflow.config import (
        disable_system_metrics_logging,
        enable_system_metrics_logging,
        get_registry_uri,
        set_registry_uri,
        set_system_metrics_node_id,
        set_system_metrics_samples_before_logging,
        set_system_metrics_sampling_interval,
    )
    from mlflow.models.evaluation.deprecated import evaluate
    from mlflow.models.evaluation.validation import validate_evaluation_results
    from mlflow.projects import run
    from mlflow.tracking._model_registry.fluent import (
        # TODO: Prompt Registry APIs are moved to the `mlflow.genai` namespace and direct
        # imports from mlflow will be deprecated in the future.
        delete_prompt_alias,
        load_prompt,
        register_model,
        register_prompt,
        search_model_versions,
        search_prompts,
        search_registered_models,
        set_model_version_tag,
        set_prompt_alias,
    )
    from mlflow.tracking.fluent import (
        ActiveModel,
        ActiveRun,
        autolog,
        clear_active_model,
        create_experiment,
        create_external_model,
        delete_experiment,
        delete_logged_model_tag,
        delete_run,
        delete_tag,
        end_run,
        finalize_logged_model,
        flush_artifact_async_logging,
        flush_async_logging,
        get_active_model_id,
        get_artifact_uri,
        get_experiment,
        get_experiment_by_name,
        get_logged_model,
        get_parent_run,
        get_run,
        initialize_logged_model,
        last_active_run,
        last_logged_model,
        load_table,
        log_artifact,
        log_artifacts,
        log_dict,
        log_figure,
        log_image,
        log_input,
        log_inputs,
        log_metric,
        log_metrics,
        log_model_params,
        log_outputs,
        log_param,
        log_params,
        log_table,
        log_text,
        search_experiments,
        search_logged_models,
        search_runs,
        set_active_model,
        set_experiment_tag,
        set_experiment_tags,
        set_logged_model_tags,
        set_tag,
        set_tags,
        start_run,
    )
    from mlflow.tracking.multimedia import Image
    from mlflow.utils.async_logging.run_operations import RunOperations  # noqa: F401
    from mlflow.utils.credentials import login
    from mlflow.utils.doctor import doctor

    __all__ += [
        "ActiveRun",
        "ActiveModel",
        "MlflowClient",
        "MlflowException",
        "autolog",
        "clear_active_model",
        "create_experiment",
        "create_external_model",
        "delete_experiment",
        "delete_run",
        "delete_tag",
        "disable_system_metrics_logging",
        "doctor",
        "enable_system_metrics_logging",
        "end_run",
        "evaluate",
        "finalize_logged_model",
        "flush_async_logging",
        "flush_artifact_async_logging",
        "get_active_model_id",
        "get_artifact_uri",
        "get_experiment",
        "get_experiment_by_name",
        "get_logged_model",
        "get_parent_run",
        "get_registry_uri",
        "get_run",
        "initialize_logged_model",
        "last_active_run",
        "last_logged_model",
        "load_table",
        "log_artifact",
        "log_artifacts",
        "log_dict",
        "log_figure",
        "log_image",
        "log_input",
        "log_inputs",
        "log_model_params",
        "log_outputs",
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
        "search_logged_models",
        "search_model_versions",
        "search_registered_models",
        "search_runs",
        "search_prompts",
        "set_active_model",
        "set_experiment_tag",
        "set_experiment_tags",
        "set_model_version_tag",
        "set_registry_uri",
        "set_system_metrics_node_id",
        "set_system_metrics_samples_before_logging",
        "set_system_metrics_sampling_interval",
        "set_tag",
        "set_tags",
        "start_run",
        "validate_evaluation_results",
        "Image",
        # Prompt Registry APIs
        # TODO: Prompt Registry APIs are moved to the `mlflow.genai` namespace and direct
        # imports from mlflow will be deprecated in the future.
        "load_prompt",
        "register_prompt",
        "search_prompts",
        "set_prompt_alias",
        "delete_prompt_alias",
        "set_logged_model_tags",
        "delete_logged_model_tag",
    ]


# `mlflow.gateway` depends on optional dependencies such as pydantic, psutil, and has version
# restrictions for dependencies. Importing this module fails if they are not installed or
# if invalid versions of these required packages are installed.
with contextlib.suppress(Exception):
    from mlflow import gateway  # noqa: F401

    __all__.append("gateway")
