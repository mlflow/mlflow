from typing import List

_INFERENCE_TASK_COMPLETIONS = "llm/v1/completions"
_INFERENCE_TASK_CHAT = "llm/v1/chat"


def _get_output_and_usage_from_tensor(
    prompt: str, output_tensor: List[int], pipeline, model_config, inference_task
):
    """
    Decode the output tensor and return the output text and usage information as a dictionary.
    """
    usage = _get_token_usage(prompt, output_tensor, pipeline, model_config)
    completions_text = _get_completions_text(prompt, output_tensor, pipeline)
    finish_reason = _get_finish_reason(
        usage["total_tokens"], usage["completion_tokens"], model_config
    )

    output_dict = {"finish_reason": finish_reason, "usage": usage}

    if inference_task == _INFERENCE_TASK_COMPLETIONS:
        output_dict["text"] = completions_text
    elif inference_task == _INFERENCE_TASK_CHAT:
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
