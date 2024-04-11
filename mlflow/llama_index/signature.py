import logging

from mlflow.environment_variables import MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema
from mlflow.utils.process import _IS_UNIX
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout

_logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = ParamSchema(
    [
        ParamSpec(  # PLACEHOLDER: needs more thought
            name="engine",
            dtype="string",
            default="query",
        )
    ]
)
_TEXT2TEXT_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string")]),
    outputs=Schema([ColSpec("string")]),
    params=_DEFAULT_PARAMS,
)

_DEFAULT_SIGNATURE = {"VectorStoreIndex": _TEXT2TEXT_SIGNATURE}


def _signature_has_required_params(signature: ModelSignature) -> bool:
    """
    PLACEHOLDER: validate that the signature supports the relevant inference type
    """
    if signature.params and signature.params["engine_type"]:
        return True


def generate_signature_output(index, data, model_config=None, flavor_config=None, params=None):
    # Lazy import to avoid circular dependencies. Ideally we should move _LlamaIndexModelWrapper
    # out from __init__.py to avoid this.
    from mlflow.llama_index import _LlamaIndexModelWrapper

    return _LlamaIndexModelWrapper(
        index=index, model_config=model_config, flavor_config=flavor_config
    ).predict(data, params=params)


def _infer_signature_with_example(
    index, example, model_config=None, flavor_config=None, timeout=None
) -> ModelSignature:
    params = None
    if _contains_params(example):
        example, params = example

    if "engine" not in params:
        params["engine"] = "query"  # TODO: map default to an object that supports

    # example = format_input_example_for_special_cases(example, pipeline)

    if timeout:
        _logger.info(
            "Running model prediction to infer the model output signature with a timeout "
            f"of {timeout} seconds. You can specify a different timeout by setting the "
            f"environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}."
        )
        with run_with_timeout(timeout):
            prediction = generate_signature_output(
                index, example, model_config, flavor_config, params
            )
    else:
        prediction = generate_signature_output(index, example, model_config, flavor_config, params)
    return infer_signature(example, prediction, params)


def infer_signature_from_input_example(
    index, example=None, model_config=None, flavor_config=None
) -> ModelSignature:
    """ """
    if example is not None:
        try:
            timeout = MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT.get()
            if timeout and not _IS_UNIX:
                timeout = None
                _logger.warning(
                    "On Windows, timeout is not supported for model signature inference. "
                    "Therefore, the operation is not bound by a timeout and may hang indefinitely. "
                    "If it hangs, please consider specifying the signature manually."
                )
            return _infer_signature_with_example(
                index, example, model_config, flavor_config, timeout
            )
        except Exception as e:
            if isinstance(e, MlflowTimeoutError):
                msg = (
                    "Attempted to generate a signature for the saved pipeline but prediction timed "
                    f"out after {timeout} seconds. Falling back to the default signature for the "
                    "pipeline. You can specify a signature manually or increase the timeout "
                    f"by setting the environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}"
                )
            else:
                msg = (
                    "Attempted to generate a signature for the saved pipeline but encountered an "
                    f"error. Fall back to the default signature for the pipeline type. Error: {e}"
                )
            _logger.warning(msg)

    import llama_index.core

    for index_type, signature in _DEFAULT_SIGNATURE.items():
        if isinstance(index, getattr(llama_index.core, index_type)):
            return signature

    _logger.warning(
        "An unsupported llama index object was supplied for signature inference. Either provide an "
        "`input_example` or generate a signature manually via `infer_signature` to have a "
        "signature recorded in the MLmodel file."
    )


def validate_and_resolve_signature(signature: ModelSignature) -> ModelSignature:
    """
    Ensure that a signature is valid based on inference method. If invalid, resolve the issue.
    """
    if not _signature_has_required_params(signature):
        signature.params = _DEFAULT_PARAMS

    return signature
