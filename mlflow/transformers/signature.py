import json
import logging

import numpy as np

from mlflow.environment_variables import MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.types.schema import ColSpec, DataType, Schema, TensorSpec
from mlflow.utils.os import is_windows
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout

_logger = logging.getLogger(__name__)


_TEXT2TEXT_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string")]),
    outputs=Schema([ColSpec("string")]),
)
_CLASSIFICATION_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string")]),
    outputs=Schema([ColSpec("string", name="label"), ColSpec("double", name="score")]),
)

# Order is important here, the first matching task type will be used
_DEFAULT_SIGNATURE_FOR_TASK = {
    "token-classification": _TEXT2TEXT_SIGNATURE,
    "translation": _TEXT2TEXT_SIGNATURE,
    "text-generation": _TEXT2TEXT_SIGNATURE,
    "text2text-generation": _TEXT2TEXT_SIGNATURE,
    "text-classification": _CLASSIFICATION_SIGNATURE,
    "conversational": _TEXT2TEXT_SIGNATURE,
    "fill-mask": _TEXT2TEXT_SIGNATURE,
    "summarization": _TEXT2TEXT_SIGNATURE,
    "image-classification": _CLASSIFICATION_SIGNATURE,
    "zero-shot-classification": ModelSignature(
        inputs=Schema(
            [
                ColSpec(DataType.string, name="sequences"),
                ColSpec(DataType.string, name="candidate_labels"),
                ColSpec(DataType.string, name="hypothesis_template"),
            ]
        ),
        outputs=Schema(
            [
                ColSpec(DataType.string, name="sequence"),
                ColSpec(DataType.string, name="labels"),
                ColSpec(DataType.double, name="scores"),
            ]
        ),
    ),
    "automatic-speech-recognition": ModelSignature(
        inputs=Schema([ColSpec(DataType.binary)]),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    "audio-classification": ModelSignature(
        inputs=Schema([ColSpec(DataType.binary)]),
        outputs=Schema(
            [ColSpec(DataType.double, name="score"), ColSpec(DataType.string, name="label")]
        ),
    ),
    "table-question-answering": ModelSignature(
        inputs=Schema(
            [ColSpec(DataType.string, name="query"), ColSpec(DataType.string, name="table")]
        ),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    "question-answering": ModelSignature(
        inputs=Schema(
            [ColSpec(DataType.string, name="question"), ColSpec(DataType.string, name="context")]
        ),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    "feature-extraction": ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]),
        outputs=Schema([TensorSpec(np.dtype("float64"), [-1], "double")]),
    ),
}


def infer_or_get_default_signature(
    pipeline, example=None, model_config=None, flavor_config=None
) -> ModelSignature:
    """
    Assigns a default ModelSignature for a given Pipeline type that has pyfunc support. These
    default signatures should only be generated and assigned when saving a model iff the user
    has not supplied a signature.
    For signature inference in some Pipelines that support complex input types, an input example
    is needed.
    """
    import transformers

    if example is not None and isinstance(pipeline, transformers.Pipeline):
        try:
            timeout = MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT.get()
            if timeout and is_windows():
                timeout = None
                _logger.warning(
                    "On Windows, timeout is not supported for model signature inference. "
                    "Therefore, the operation is not bound by a timeout and may hang indefinitely. "
                    "If it hangs, please consider specifying the signature manually."
                )
            return _infer_signature_with_example(
                pipeline, example, model_config, flavor_config, timeout
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

    task = getattr(pipeline, "task", None)
    if task.startswith("translation_"):
        task = "translation"
    if signature := _DEFAULT_SIGNATURE_FOR_TASK.get(task):
        return signature

    _logger.warning(
        "An unsupported task type was supplied for signature inference. Either provide an "
        "`input_example` or generate a signature manually via `infer_signature` to have a "
        "signature recorded in the MLmodel file."
    )


def _infer_signature_with_example(
    pipeline, example, model_config=None, flavor_config=None, timeout=None
) -> ModelSignature:
    params = None
    if _contains_params(example):
        example, params = example
    example = format_input_example_for_special_cases(example, pipeline)

    if timeout:
        _logger.info(
            "Running model prediction to infer the model output signature with a timeout "
            f"of {timeout} seconds. You can specify a different timeout by setting the "
            f"environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}."
        )
        with run_with_timeout(timeout):
            prediction = generate_signature_output(
                pipeline, example, model_config, flavor_config, params
            )
    else:
        prediction = generate_signature_output(
            pipeline, example, model_config, flavor_config, params
        )
    return infer_signature(example, prediction, params)


def format_input_example_for_special_cases(input_example, pipeline):
    """
    Handles special formatting for specific types of Pipelines so that the displayed example
    reflects the correct example input structure that mirrors the behavior of the input parsing
    for pyfunc.
    """
    input_data = input_example[0] if isinstance(input_example, tuple) else input_example

    if (
        pipeline.task == "zero-shot-classification"
        and isinstance(input_data, dict)
        and isinstance(input_data["candidate_labels"], list)
    ):
        input_data["candidate_labels"] = json.dumps(input_data["candidate_labels"])
    return input_data if not isinstance(input_example, tuple) else (input_data, input_example[1])


def generate_signature_output(pipeline, data, model_config=None, flavor_config=None, params=None):
    # Lazy import to avoid circular dependencies. Ideally we should move _TransformersWrapper
    # out from __init__.py to avoid this.
    from mlflow.transformers import _TransformersWrapper

    return _TransformersWrapper(
        pipeline=pipeline, model_config=model_config, flavor_config=flavor_config
    ).predict(data, params=params)
