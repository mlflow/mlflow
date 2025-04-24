import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy

from packaging.version import Version

import mlflow
from mlflow.entities import RunTag
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.langchain.runnables import get_runnable_steps
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import name_utils
from mlflow.utils.autologging_utils import get_autologging_config
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

_logger = logging.getLogger(__name__)

UNSUPPORTED_LOG_MODEL_MESSAGE = (
    "MLflow autologging does not support logging models containing BaseRetriever because "
    "logging the model requires `loader_fn` and `persist_dir`. Please log the model manually "
    "using `mlflow.langchain.log_model(model, artifact_path, loader_fn=..., persist_dir=...)`"
)
INFERENCE_FILE_NAME = "inference_inputs_outputs.json"


# A *global* state that indicates whether MLflow should patch the inference method
# for artifact auto-logging (model, signature input example). This disablement
# is global across threads, as single model inference can trigger multiple threads,
# for example, LangChain's batch()/abatch() API processes each request in a child thread.
IS_PATCHING_DISABLED_FOR_ARTIFACTS = False


@contextlib.contextmanager
def disable_patching():
    """
    Temporarily disable auto-logging for optional artifacts (model, signature, input
    examples) to avoid "double-logging" when invoking the patched chain. Without this
    disablement applied, the patched inference method calls child components that may
    also be patched, leading to redundant logging.
    """
    global IS_PATCHING_DISABLED_FOR_ARTIFACTS
    original_artifact_flag = IS_PATCHING_DISABLED_FOR_ARTIFACTS
    IS_PATCHING_DISABLED_FOR_ARTIFACTS = True

    try:
        yield
    finally:
        IS_PATCHING_DISABLED_FOR_ARTIFACTS = original_artifact_flag


def patched_inference(func_name, original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables
    logging the traces, and other optional artifacts like model, input examples, etc.

    We patch inference functions for different models based on their usage.
    """

    def _invoke(self, *args, **kwargs):
        with disable_patching():
            return original(self, *args, **kwargs)

    config = AutoLoggingConfig.init(mlflow.langchain.FLAVOR_NAME)
    if not IS_PATCHING_DISABLED_FOR_ARTIFACTS and config.should_log_optional_artifacts():
        with _setup_autolog_run(config, self) as run_id:
            result = _invoke(self, *args, **kwargs)
            _log_optional_artifacts(config, run_id, result, self, func_name, *args, **kwargs)
    else:
        result = _invoke(self, *args, **kwargs)
    return result


@contextlib.contextmanager
def _setup_autolog_run(config, model):
    """Set up autologging run and return the run ID.

    This function only creates a run when there is no active run and the model does not have
    a run ID attribute propagated from the previous call. Iff it creates a new run, MLflow should
    terminate the run at the end of the inference.

    Args:
        config: AutoLoggingConfig: The autologging configuration.
        model: Any: The LangChain model instance that runs the inference.

    Returns: yields the run IDs
    """
    if propagated_run_id := getattr(model, "run_id", None):
        # When model has "run_id" attribute, it means the model is already invoked once with autolog
        # enabled and the run_id is propagated from the previous call, so we don't create a new run.
        run_id = propagated_run_id
        # The run should be already terminated at the end of the previous call.
        should_terminate_run = False

    elif active_run := mlflow.active_run():
        run_id = active_run.info.run_id
        tags = _resolve_tags(config.extra_tags, active_run)
        mlflow.MlflowClient().log_batch(run_id, tags=[RunTag(k, str(v)) for k, v in tags.items()])
        should_terminate_run = False
    else:
        from mlflow.tracking.fluent import _get_experiment_id

        run = mlflow.MlflowClient().create_run(
            experiment_id=_get_experiment_id(),
            run_name="langchain-" + name_utils._generate_random_name(),
            tags=_resolve_tags(config.extra_tags),
        )
        run_id = run.info.run_id
        should_terminate_run = True

    run_status = None
    try:
        yield run_id
    except Exception:
        run_status = RunStatus.to_string(RunStatus.FAILED)
        raise
    finally:
        if should_terminate_run:
            mlflow.MlflowClient().set_terminated(run_id, status=run_status)


def _resolve_tags(extra_tags, active_run=None):
    resolved_tags = context_registry.resolve_tags(extra_tags)
    tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, resolved_tags)
    if active_run:
        # Some context tags like mlflow.runName are immutable once logged, but they might be already
        # set when the run is created, then we should avoid updating them.
        excluded_tags = {tag for tag in active_run.data.tags.keys() if tag.startswith("mlflow.")}
        tags = {k: v for k, v in tags.items() if k not in excluded_tags}
    return tags


def _get_input_data_from_function(func_name, model, args, kwargs):
    func_param_name_mapping = {
        "invoke": "input",
        "batch": "inputs",
        "stream": "input",
    }
    input_example_exc = None
    if param_name := func_param_name_mapping.get(func_name):
        inference_func = getattr(model, func_name)
        # A guard to make sure `param_name` is the first argument of inference function
        if next(iter(inspect.signature(inference_func).parameters.keys())) != param_name:
            input_example_exc = MlflowException(
                "Inference function signature changes, please contact MLflow team to "
                "fix langchain autologging.",
            )
        else:
            return args[0] if len(args) > 0 else kwargs.get(param_name)
    else:
        input_example_exc = MlflowException(
            f"Unsupported inference function. Only support {list(func_param_name_mapping.keys())}."
        )
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__} "
        f"due to {input_example_exc}."
    )


def _convert_data_to_dict(data, key):
    if isinstance(data, dict):
        return {f"{key}-{k}": v for k, v in data.items()}
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, str):
        return {key: [data]}
    raise MlflowException("Unsupported data type.")


def _update_langchain_model_config(model):
    # Langchain models are Pydantic models, and the value for extra is
    # ignored, we need to set it to allow so as to set attributes on
    # the model to keep track of logging status
    import langchain

    try:
        # LangChain 0.3.0 and above is fully migrated to Pydantic v2
        if Version(langchain.__version__) >= Version("0.3.0"):
            if hasattr(model, "model_config") and model.model_config is not None:
                model.model_config["extra"] = "allow"
                model.__pydantic_extra__ = {}
                return True
        else:
            from langchain_core.pydantic_v1 import Extra

            if hasattr(model, "__config__"):
                model.__config__.extra = Extra.allow
            return True
    except Exception as e:
        warnings.warn(
            "Failed to set extra attribute on the model for keeping track of logging status. "
            f"MLflow langchain autologging might log model several times. Error: {e}"
        )
        return False


def _runnable_with_retriever(model):
    from langchain.schema import BaseRetriever

    with contextlib.suppress(ImportError):
        from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
        from langchain.schema.runnable.passthrough import RunnableAssign

        if isinstance(model, RunnableBranch):
            return any(_runnable_with_retriever(runnable) for _, runnable in model.branches)

        if isinstance(model, RunnableParallel):
            return any(
                _runnable_with_retriever(runnable)
                for runnable in get_runnable_steps(model).values()
            )

        if isinstance(model, RunnableSequence):
            return any(_runnable_with_retriever(runnable) for runnable in get_runnable_steps(model))

        if isinstance(model, RunnableAssign):
            return _runnable_with_retriever(model.mapper)

    return isinstance(model, BaseRetriever)


def _chain_with_retriever(model):
    with contextlib.suppress(ImportError):
        from langchain.chains import RetrievalQA

        return isinstance(model, RetrievalQA)
    return False


def _log_optional_artifacts(autolog_config, run_id, result, self, func_name, *args, **kwargs):
    input_example = None
    if autolog_config.log_models and not hasattr(self, "_mlflow_model_logged"):
        if _runnable_with_retriever(self) or _chain_with_retriever(self):
            _logger.info(UNSUPPORTED_LOG_MODEL_MESSAGE)
        else:
            # warn user in case we did't capture some cases where retriever is used
            warnings.warn(UNSUPPORTED_LOG_MODEL_MESSAGE)
            if autolog_config.log_input_examples:
                input_example = deepcopy(
                    _get_input_data_from_function(func_name, self, args, kwargs)
                )
                if not autolog_config.log_model_signatures:
                    _logger.info(
                        "Signature is automatically generated for logged model if "
                        "input_example is provided. To disable log_model_signatures, "
                        "please also disable log_input_examples."
                    )

            registered_model_name = get_autologging_config(
                mlflow.langchain.FLAVOR_NAME, "registered_model_name", None
            )
            try:
                with disable_patching():
                    mlflow.langchain.log_model(
                        self,
                        "model",
                        input_example=input_example,
                        registered_model_name=registered_model_name,
                        run_id=run_id,
                    )
            except Exception as e:
                _logger.warning(f"Failed to log model due to error {e}.")
        # only try logging model once, even if it can't be logged
        # we don't want to spam the user with warnings/infos
        if _update_langchain_model_config(self):
            self._mlflow_model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if _update_langchain_model_config(self):
        # NB: We have to set these attributes AFTER the model is logged, otherwise those extra
        # attributes will be logged as a part of the pickled model and pollute the loaded model.
        if not hasattr(self, "run_id"):
            self.run_id = run_id
        if not hasattr(self, "session_id"):
            self.session_id = uuid.uuid4().hex
        self.inference_id = getattr(self, "inference_id", 0) + 1

    return result
