from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.transformers.flavor_config import FlavorKey
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    COMPLETIONS_MODEL_INPUT_SCHEMA,
    COMPLETIONS_MODEL_OUTPUT_SCHEMA,
    EMBEDDING_MODEL_INPUT_SCHEMA,
    EMBEDDING_MODEL_OUTPUT_SCHEMA,
)

if TYPE_CHECKING:
    import torch

_LLM_INFERENCE_TASK_KEY = "inference_task"
# The LLM inference task is saved as "task" in the metadata for forward compatibility with
# future Databricks Provisioned Throughput support of more model architectures for inference.
_METADATA_LLM_INFERENCE_TASK_KEY = "task"

_LLM_INFERENCE_TASK_PREFIX = "llm/v1"
_LLM_INFERENCE_TASK_COMPLETIONS = f"{_LLM_INFERENCE_TASK_PREFIX}/completions"
_LLM_INFERENCE_TASK_CHAT = f"{_LLM_INFERENCE_TASK_PREFIX}/chat"
_LLM_INFERENCE_TASK_EMBEDDING = f"{_LLM_INFERENCE_TASK_PREFIX}/embeddings"

_LLM_V1_EMBEDDING_INPUT_KEY = "input"


_LLM_INFERENCE_OBJECT_NAME = {
    _LLM_INFERENCE_TASK_COMPLETIONS: "text_completion",
    _LLM_INFERENCE_TASK_CHAT: "chat.completion",
}

_SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK = {
    "text-generation": [_LLM_INFERENCE_TASK_COMPLETIONS, _LLM_INFERENCE_TASK_CHAT],
    "feature-extraction": [_LLM_INFERENCE_TASK_EMBEDDING],
}

_SIGNATURE_FOR_LLM_INFERENCE_TASK = {
    _LLM_INFERENCE_TASK_CHAT: ModelSignature(
        inputs=CHAT_MODEL_INPUT_SCHEMA, outputs=CHAT_MODEL_OUTPUT_SCHEMA
    ),
    _LLM_INFERENCE_TASK_COMPLETIONS: ModelSignature(
        inputs=COMPLETIONS_MODEL_INPUT_SCHEMA, outputs=COMPLETIONS_MODEL_OUTPUT_SCHEMA
    ),
    _LLM_INFERENCE_TASK_EMBEDDING: ModelSignature(
        inputs=EMBEDDING_MODEL_INPUT_SCHEMA, outputs=EMBEDDING_MODEL_OUTPUT_SCHEMA
    ),
}

_LLM_INFERENCE_TASK_TO_DATA_FIELD = {
    _LLM_INFERENCE_TASK_CHAT: "messages",
    _LLM_INFERENCE_TASK_COMPLETIONS: "prompt",
}


def infer_signature_from_llm_inference_task(
    inference_task: str, signature: ModelSignature | None = None
) -> ModelSignature:
    """
    Infers the signature according to the MLflow inference task.
    Raises exception if a signature is given.
    """
    inferred_signature = _SIGNATURE_FOR_LLM_INFERENCE_TASK[inference_task]

    if signature is not None and signature != inferred_signature:
        raise MlflowException(
            f"When `task` is specified as `{inference_task}`, the signature would "
            "be set by MLflow. Please do not set the signature."
        )
    return inferred_signature


def convert_messages_to_prompt(messages: list[dict[str, Any]], tokenizer) -> str:
    """For the Chat inference task, apply chat template to messages to create prompt.

    Args:
        messages: List of message e.g. [{"role": user, "content": xxx}, ...]
        tokenizer: The tokenizer object used for inference.

    Returns:
        The prompt string contains the messages.
    """
    if not (isinstance(messages, list) and all(isinstance(msg, dict) for msg in messages)):
        raise MlflowException(
            f"Input messages should be list of dictionaries, but got: {type(messages)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        raise MlflowException(f"Failed to apply chat template: {e}")


def preprocess_llm_inference_input(
    data: pd.DataFrame | dict[str, Any],
    params: dict[str, Any] | None = None,
    flavor_config: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    When a MLflow inference task is given, return updated `data` and `params` that
    - Extract the parameters from the input data (from the first row if passed multiple rows)
    - Replace OpenAI specific parameters with Hugging Face specific parameters, in particular
      - `max_tokens` with `max_new_tokens`
      - `stop` with `stopping_criteria`

    Args:
        data: Input data for the LLM inference task. Either a pandas DataFrame (after signature
            enforcement) or a raw dictionary payload.
        params: Optional dictionary of parameters.
        flavor_config: Optional dictionary of flavor configuration.
    """
    if isinstance(data, pd.DataFrame):
        # Pandas convert None to np.nan internally, which is not preferred
        data = data.replace(np.nan, None).to_dict(orient="list")
    elif isinstance(data, dict):
        # Convert single value to list for consistency with DataFrame
        data = {k: [v] for k, v in data.items()}
    else:
        raise MlflowException(
            "Input data for a Transformer model logged with `llm/v1/chat` or `llm/v1/completions`"
            f"task is expected to be a pandas DataFrame or a dictionary, but got: {type(data)}.",
            error_code=BAD_REQUEST,
        )

    flavor_config = flavor_config or {}
    params = params or {}

    # Extract list of input data (prompt, messages) to LLM
    task = flavor_config[_LLM_INFERENCE_TASK_KEY]
    input_col = _LLM_INFERENCE_TASK_TO_DATA_FIELD.get(task)
    if input_col not in data:
        raise MlflowException(
            f"Transformer model saved with `{task}` task excepts `{input_col}`"
            "to be passed as input data.",
            error_code=BAD_REQUEST,
        )
    update_data = data.pop(input_col)

    # The rest of fields in input payload should goes to params and override default ones
    params_in_data = {k: v[0] for k, v in data.items() if v[0] is not None}
    params = {**params, **params_in_data}

    if max_tokens := params.pop("max_tokens", None):
        params["max_new_tokens"] = max_tokens
    if stop := params.pop("stop", None):
        params["stopping_criteria"] = _get_stopping_criteria(
            stop,
            flavor_config.get(FlavorKey.MODEL_NAME),
        )
    return update_data, params


def _get_stopping_criteria(stop: str | list[str] | None, model_name: str | None = None):
    """Return a list of Hugging Face stopping criteria objects for the given stop sequences."""
    from transformers import AutoTokenizer, StoppingCriteria

    if stop is None or model_name is None:
        return None

    if isinstance(stop, str):
        stop = [stop]

    # To tokenize the stop sequences for stopping criteria, we need to use the slow tokenizer
    # for matching the actual tokens, according to https://github.com/huggingface/transformers/issues/27704
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def _get_slow_token_ids(seq: str):
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(seq))

    # NB: We need to define this as an inner class to avoid importing
    # transformers in the global scope that confuses autologging
    class _StopSequenceMatchCriteria(StoppingCriteria):
        def __init__(self, stop_sequence_ids):
            self.stop_sequence_ids = stop_sequence_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            last_ids = input_ids[:, -len(self.stop_sequence_ids) :].tolist()
            return self.stop_sequence_ids in last_ids

    stopping_criteria = []
    for stop_sequence in stop:
        # Add stopping criteria for both with and without space, such as "stopword" and " stopword"
        token_ids = _get_slow_token_ids(stop_sequence)
        token_ids_with_space = _get_slow_token_ids(" " + stop_sequence)
        stopping_criteria += [
            _StopSequenceMatchCriteria(token_ids),
            _StopSequenceMatchCriteria(token_ids_with_space),
        ]

    return stopping_criteria


def postprocess_output_for_llm_inference_task(
    data: list[str],
    output_tensors: list[list[int]],
    pipeline,
    flavor_config,
    model_config,
    inference_task,
):
    """
    Wrap output data with usage information according to the MLflow inference task.

    Example:
        .. code-block:: python
            data = ["How to learn Python in 3 weeks?"]
            output_tensors = [
                [
                    1128,
                    304,
                    ...,
                    29879,
                ]
            ]
            output_dicts = postprocess_output_for_llm_inference_task(data, output_tensors, **kwargs)

            assert output_dicts == [
                {
                    "id": "e4f3b3e3-3b3e-4b3e-8b3e-3b3e4b3e8b3e",
                    "object": "text_completion",
                    "created": 1707466970,
                    "model": "loaded_model_name",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "text": "1. Start with a beginner's",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
                }
            ]

    Args:
        data: List of text input prompts.
        output_tensors: List of output tensors that contain the generated tokens (including
            the prompt tokens) corresponding to each input prompt.
        pipeline: The pipeline object used for inference.
        flavor_config: The flavor configuration dictionary for the model.
        model_config: The model configuration dictionary used for inference.
        inference_task: The MLflow inference task.

    Returns:
        List of dictionaries containing the output text and usage information for each input prompt.
    """
    return [
        _get_output_and_usage_from_tensor(
            input_data, output_tensor, pipeline, flavor_config, model_config, inference_task
        )
        for input_data, output_tensor in zip(data, output_tensors)
    ]


def _get_output_and_usage_from_tensor(
    prompt: str, output_tensor: list[int], pipeline, flavor_config, model_config, inference_task
):
    """
    Decode the output tensor and return the output text and usage information as a dictionary
    to make the output in OpenAI compatible format.
    """
    usage = _get_token_usage(prompt, output_tensor, pipeline, model_config)
    completions_text = _get_completions_text(prompt, output_tensor, pipeline)
    finish_reason = _get_finish_reason(
        usage["total_tokens"], usage["completion_tokens"], model_config
    )

    output_dict = {
        "id": str(uuid.uuid4()),
        "object": _LLM_INFERENCE_OBJECT_NAME[inference_task],
        "created": int(time.time()),
        "model": flavor_config.get("source_model_name", ""),
        "usage": usage,
    }

    completion_choice = {
        "index": 0,
        "finish_reason": finish_reason,
    }

    if inference_task == _LLM_INFERENCE_TASK_COMPLETIONS:
        completion_choice["text"] = completions_text
    elif inference_task == _LLM_INFERENCE_TASK_CHAT:
        completion_choice["message"] = {"role": "assistant", "content": completions_text}

    output_dict["choices"] = [completion_choice]

    return output_dict


def _get_completions_text(prompt: str, output_tensor: list[int], pipeline):
    """Decode generated text from output tensor and remove the input prompt."""
    generated_text = pipeline.tokenizer.decode(
        output_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # In order to correctly remove the prompt tokens from the decoded tokens,
    # we need to acquire the length of the prompt without special tokens
    prompt_ids_without_special_tokens = pipeline.tokenizer(
        prompt, return_tensors=pipeline.framework, add_special_tokens=False
    )["input_ids"][0]

    prompt_length = len(
        pipeline.tokenizer.decode(
            prompt_ids_without_special_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )

    return generated_text[prompt_length:].lstrip()


def _get_token_usage(prompt: str, output_tensor: list[int], pipeline, model_config):
    """Return the prompt tokens, completion tokens, and the total tokens as dict."""
    inputs = pipeline.tokenizer(
        prompt,
        return_tensors=pipeline.framework,
        max_length=model_config.get("max_length", None),
        add_special_tokens=False,
    )

    prompt_tokens = inputs["input_ids"].shape[-1]
    total_tokens = len(output_tensor)
    completions_tokens = total_tokens - prompt_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completions_tokens,
        "total_tokens": total_tokens,
    }


def _get_finish_reason(total_tokens: int, completion_tokens: int, model_config):
    """Determine the reason that the text generation finished."""
    finish_reason = "stop"

    if total_tokens > model_config.get(
        "max_length", float("inf")
    ) or completion_tokens == model_config.get("max_new_tokens", float("inf")):
        finish_reason = "length"

    return finish_reason


def _get_default_task_for_llm_inference_task(llm_inference_task: str | None) -> str | None:
    """
    Get corresponding original Transformers task for the given LLM inference task.

    NB: This assumes there is only one original Transformers task for each LLM inference
      task, which might not be true in the future.
    """
    for task, llm_tasks in _SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK.items():
        if llm_inference_task in llm_tasks:
            return task
    return None


def preprocess_llm_embedding_params(
    data: pd.DataFrame | dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """
    When `llm/v1/embeddings` task is given, extract the input data (with "input" key) and
    parameters, and format the input data into the unified format for easier downstream handling.

    The handling is more complicated than other LLM inference tasks because the embedding endpoint
    accepts heterogeneous input - both string and list of strings as input. Also we don't enforce
    the input schema always, so there are 4 possible input types:
      (1)  Pandas DataFrame with string column
      (2)  Pandas DataFrame with list of strings column
      (3)  Dictionary with string value
      (4)  Dictionary with list of strings value
    In all cases, the returned input data will be a list of strings.

    Args:
        data: Input data for the embedding task.

    Returns:
        Tuple of input data and parameters dictionary.
    """
    if isinstance(data, pd.DataFrame):
        params = {}
        for col in data.columns:
            if col == _LLM_V1_EMBEDDING_INPUT_KEY:
                input_data = data[col].to_list()
                if isinstance(input_data[0], list):
                    input_data = input_data[0]
            else:
                params[col] = data[col].tolist()[0]
    else:
        # NB: Input schema is not enforced for the embedding task because of the heterogeneous
        # input type, so we have to cast the input data into unified format here.
        input_data = data.get(_LLM_V1_EMBEDDING_INPUT_KEY)
        if isinstance(input, str):
            input_data = [input_data]
        params = {k: v for k, v in data.items() if k != _LLM_V1_EMBEDDING_INPUT_KEY}

    return input_data, params


def postprocess_output_for_llm_v1_embedding_task(
    input_prompts: list[str],
    output_tensors: list[list[float]],
    tokenizer,
):
    """
    Wrap output data with usage information.

    Examples:
        .. code-block:: python
            input_prompt = ["hello world and hello mlflow"]
            output_embedding = [0.47137904, 0.4669448, ..., 0.69726706]
            output_dicts = postprocess_output_for_llm_v1_embedding_task(
                input_prompt, output_embedding
            )
            assert output_dicts == [
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.47137904, 0.4669448, ..., 0.69726706],
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            ]

    Args:
        input_prompts: text input prompts
        output_tensors: List of output tensors that contain the generated embeddings
        tokenizer: The tokenizer object used for inference.

    Returns:
            Dictionaries containing the output embedding and usage information for each
            input prompt.
    """
    prompt_tokens = sum(len(tokenizer(prompt)["input_ids"]) for prompt in input_prompts)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": tensor,
            }
            for i, tensor in enumerate(output_tensors)
        ],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
