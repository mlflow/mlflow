import contextlib
import inspect
import logging
import warnings
from copy import deepcopy

import langchain
from packaging.version import Version
from pydantic.v1.config import Extra

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
    disable_autologging,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["maximum"])

_lc_version = Version(langchain.__version__)
_logger = logging.getLogger(__name__)


def _get_input_data_from_invoke(model, args, kwargs):
    from langchain.schema.runnable import Runnable

    invoke_func = model.invoke
    input_example_exc = None
    try:
        if isinstance(model, Runnable):
            # A guard to make sure `input` is the first argument of `invoke` function
            assert next(iter(inspect.signature(invoke_func).parameters.keys())) == "input"
            return kwargs["input"] if "input" in kwargs else args[0]
    except Exception as e:
        input_example_exc = e
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__}"
        + f" due to {input_example_exc}."
        if input_example_exc
        else ""
    )


def _combine_input_and_output(input, output):
    """
    Combine input and output into a single dictionary
    """
    if input is None:
        if isinstance(output, dict):
            return output
        return {"output": output if isinstance(output, list) else [output]}
    if isinstance(input, (str, dict)):
        return {"input": [input], "output": [output]}
    if isinstance(input, list) and (
        all(isinstance(x, str) for x in input) or all(isinstance(x, dict) for x in input)
    ):
        if not isinstance(output, list) or len(output) != len(input):
            raise MlflowException(
                "Failed to combine input and output data with different lengths "
                "into a single pandas DataFrame. "
            )
        return {"input": input, "output": output}
    raise MlflowException("Unsupported input type.")


@contextlib.contextmanager
def _wrap_func_with_run(run_id, **kwargs):
    if mlflow.active_run():
        yield
    else:
        with mlflow.start_run(run_id=run_id, **kwargs):
            yield


def patched_invoke(original, self, *args, **kwargs):
    """
    A patched implementation of langchain runnables `invoke` which enables logging the
    following parameters, metrics and artifacts:

    - model
    - model parameters
    - invoke data
    """
    from langchain.callbacks import MlflowCallbackHandler
    from langchain.schema.runnable.config import RunnableConfig

    class _MLflowLangchainCallback(MlflowCallbackHandler, metaclass=ExceptionSafeAbstractClass):
        """
        Callback for auto-logging metrics and parameters.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    if not MIN_REQ_VERSION <= _lc_version <= MAX_REQ_VERSION:
        warnings.warn(
            "Autologging is known to be compatible with langchain versions between "
            f"{MIN_REQ_VERSION} and {MAX_REQ_VERSION} and may not succeed with packages "
            "outside this range."
        )

    config = kwargs.get("config", None)
    run_id = self.run_id if hasattr(self, "run_id") else None
    if active_run := mlflow.active_run():
        if run_id is None:
            run_id = active_run.info.run_id
        else:
            if run_id != active_run.info.run_id:
                raise MlflowException(
                    "Please end current run when autologging is on "
                    "because we need to use the run attached to current "
                    "model instance for logging."
                )
    # TODO: test adding callbacks works
    mlflow_callback = _MLflowLangchainCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=run_id,
    )
    if config is None:
        callbacks = [mlflow_callback]
        config = RunnableConfig(callbacks=callbacks)
    else:
        callbacks = config.get("callbacks", [])
        callbacks.append(mlflow_callback)
        config["callbacks"] = callbacks
    kwargs["config"] = config
    with disable_autologging():
        result = original(self, *args, **kwargs)

    mlflow_callback.flush_tracker()

    log_models = get_autologging_config(mlflow.langchain.FLAVOR_NAME, "log_models", False)
    log_input_examples = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_input_examples", False
    )
    log_model_signatures = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_model_signatures", False
    )
    input_example = None
    if log_models and not hasattr(self, "model_logged"):
        if log_input_examples:
            input_example = deepcopy(_get_input_data_from_invoke(self, args, kwargs))
            if not log_model_signatures:
                _logger.warning(
                    "Signature is automatically generated for logged model if "
                    "input_example is provided. To disable log_model_signatures, "
                    "please also disable log_input_examples."
                )

        registered_model_name = get_autologging_config(
            mlflow.langchain.FLAVOR_NAME, "registered_model_name", None
        )
        extra_tags = get_autologging_config(mlflow.langchain.FLAVOR_NAME, "extra_tags", None)
        tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, extra_tags)
        # self manage the run as we need to get the run_id from mlflow_callback
        # only log the tags once the first time we log the model
        with _wrap_func_with_run(mlflow_callback.mlflg.run.info.run_id, tags=tags):
            mlflow.langchain.log_model(
                self,
                "model",
                input_example=input_example,
                registered_model_name=registered_model_name,
            )
        if hasattr(self, "__config__"):
            self.__config__.extra = Extra.allow
        self.model_logged = True

    # Even if the model is not logged, we keep a single
    # run per model
    if hasattr(self, "__config__"):
        self.__config__.extra = Extra.allow
    self.run_id = mlflow_callback.mlflg.run.info.run_id

    log_inference_history = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_inference_history", False
    )
    if log_inference_history:
        if input_example is None:
            input_data = deepcopy(_get_input_data_from_invoke(self, args, kwargs))
            if input_data is None:
                _logger.warning("Input data gathering failed, only log inference results.")
        else:
            input_data = input_example
        data_dict = _combine_input_and_output(input_data, result)
        with _wrap_func_with_run(mlflow_callback.mlflg.run.info.run_id):
            mlflow.log_table(data_dict, "inference_history.json")

    return result
