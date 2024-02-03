from typing import List
from unittest import mock

import torch

from mlflow.transformers.llm_inference_utils import (
    _get_finish_reason,
    _get_output_and_usage_from_tensor,
    _get_token_usage,
)


class DummyTokenizer:
    def __call__(self, text: str, **kwargs):
        input_ids = list(map(int, text.split(" ")))
        return {"input_ids": torch.tensor(input_ids)}

    def decode(self, tensor: List[int], **kwargs):
        return " ".join([str(x) for x in tensor])


def test_output_dict_for_completions():
    prompt = "1 2 3"
    output_tensor = [1, 2, 3, 4, 5]
    model_config = {"max_new_tokens": 2}
    inference_task = "llm/v1/completions"

    pipeline = mock.MagicMock()
    pipeline.tokenizer = DummyTokenizer()

    output_dict = _get_output_and_usage_from_tensor(
        prompt, output_tensor, pipeline, model_config, inference_task
    )

    assert output_dict["text"] == "4 5"
    assert output_dict["finish_reason"] == "length"

    usage = output_dict["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_token_usage():
    prompt = "1 2 3"
    output_tensor = [1, 2, 3, 4, 5]

    pipeline = mock.MagicMock()
    pipeline.tokenizer = DummyTokenizer()

    usage = _get_token_usage(prompt, output_tensor, pipeline, {})
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
