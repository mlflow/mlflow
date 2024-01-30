from unittest import mock

import torch

from mlflow.transformers.llm_inference_utils import (
    _get_finish_reason,
    _get_output_and_usage_from_tensor,
    _get_token_usage,
)


def test_output_dict_for_completions(text_generation_pipeline):
    prompt = "a b c"
    output_tensor = [1, 2, 3, 4, 5]
    model_config = {"max_new_tokens": 2}
    inference_task = "llm/v1/completions"

    text_generation_pipeline.tokenizer = mock.MagicMock(
        return_value={"input_ids": torch.tensor([1, 2, 3])}
    )
    text_generation_pipeline.tokenizer.decode = mock.MagicMock(return_value="a b c d e")

    output_dict = _get_output_and_usage_from_tensor(
        prompt, output_tensor, text_generation_pipeline, model_config, inference_task
    )

    assert output_dict["text"] == "d e"
    assert output_dict["finish_reason"] == "length"

    usage = output_dict["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_token_usage(text_generation_pipeline):
    prompt = "one two three"
    output_tensor = [1, 2, 3, 4, 5]

    text_generation_pipeline.tokenizer = mock.MagicMock(
        return_value={"input_ids": torch.tensor([1, 2, 3])}
    )

    usage = _get_token_usage(prompt, output_tensor, text_generation_pipeline, {})
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 5


def test_finish_reason():
    assert _get_finish_reason(total_tokens=20, completion_tokens=10, model_config={}) == "stop"

    assert (
        _get_finish_reason(
            total_tokens=20, completion_tokens=10, model_config={"max_new_tokens": 10}
        )
        == "length"
    )

    assert (
        _get_finish_reason(total_tokens=20, completion_tokens=10, model_config={"max_length": 15})
        == "length"
    )
