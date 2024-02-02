from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, StoppingCriteria

_LLM_INFERENCE_TASK_KEY = "inference_task"
# The LLM inference task is saved as "task" in the metadata for forward compatibility with
# future Databricks Provisioned Throughput support of more model architectures for inference.
_METADATA_LLM_INFERENCE_TASK_KEY = "task"

_LLM_INFERENCE_TASK_COMPLETIONS = "llm/v1/completions"
_LLM_INFERENCE_TASK_CHAT = "llm/v1/chat"

_SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK = {
    "text-generation": [_LLM_INFERENCE_TASK_COMPLETIONS, _LLM_INFERENCE_TASK_CHAT],
}


def preprocess_llm_inference_params(
    params: Optional[Dict[str, Any]] = None, flavor_config: Optional[Dict[str, Any]] = None
):
    """Replace OpenAI specific parameters with Hugging Face specific parameters."""
    if params is None:
        return

    params["max_new_tokens"] = params.pop("max_tokens", None)

    model_name = None
    if flavor_config is not None:
        model_name = flavor_config.get("source_model_name", None)
    params["stopping_criteria"] = _set_stopping_criteria(params.pop("stop", None), model_name)


def _set_stopping_criteria(stop: Optional[Union[str, List[str]]], model_name: Optional[str] = None):
    if stop is None or model_name is None:
        return None

    if isinstance(stop, str):
        stop = [stop]

    class StopSequenceMatchCriteria(StoppingCriteria):
        def __init__(self, stop_sequence_ids):
            self.stop_sequence_ids = stop_sequence_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            last_ids = input_ids[:, -len(self.stop_sequence_ids) :].tolist()
            return self.stop_sequence_ids in last_ids

    # To tokenize the stop sequences for stopping criteria, we need to use the slow tokenizer
    # for matching the actual tokens, according to https://github.com/huggingface/transformers/issues/27704
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    get_token_ids = lambda seq: tokenizer.convert_tokens_to_ids(tokenizer._tokenize(seq))

    stopping_criteria = []
    for stop_sequence in stop:
        # Add stopping criteria for both with and without space, such as "stopword" and " stopword"
        token_ids = get_token_ids(stop_sequence)
        token_ids_with_space = get_token_ids(" " + stop_sequence)
        stopping_criteria += [
            StopSequenceMatchCriteria(token_ids),
            StopSequenceMatchCriteria(token_ids_with_space),
        ]

    return stopping_criteria


def postprocess_output_for_llm_inference_task(
    data: List[str], output_tensors: List[List[int]], pipeline, model_config, inference_task
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
                    "text": "1. Start with a beginner's",
                    "finish_reason": "length",
                    "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
                }
            ]

    Args:
        data: List of text input prompts.
        output_tensors: List of output tensors that contain the generated tokens (including
            the prompt tokens) corresponding to each input prompt.
        pipeline: The pipeline object used for inference.
        model_config: The model configuration dictionary used for inference.
        inference_task: The MLflow inference task.

    Returns:
        List of dictionaries containing the output text and usage information for each input prompt.
    """
    output_dicts = []
    for input_data, output_tensor in zip(data, output_tensors):
        output_dict = _get_output_and_usage_from_tensor(
            input_data, output_tensor, pipeline, model_config, inference_task
        )
        output_dicts.append(output_dict)

    return output_dicts


def _get_output_and_usage_from_tensor(
    prompt: str, output_tensor: List[int], pipeline, model_config, inference_task
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

    output_dict = {"finish_reason": finish_reason, "usage": usage}

    if inference_task == _LLM_INFERENCE_TASK_COMPLETIONS:
        output_dict["text"] = completions_text
    elif inference_task == _LLM_INFERENCE_TASK_CHAT:
        output_dict["messages"] = {"role": "assistant", "content": completions_text}

    return output_dict


def _get_completions_text(prompt: str, output_tensor: List[int], pipeline):
    """Decode generated text from output tensor and remove the input prompt."""
    generated_text = pipeline.tokenizer.decode(
        output_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return generated_text[len(prompt) :].lstrip()


def _get_token_usage(prompt: str, output_tensor: List[int], pipeline, model_config):
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
