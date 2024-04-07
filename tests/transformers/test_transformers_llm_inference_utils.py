import uuid
from typing import Dict, List
from unittest import mock

import pandas as pd
import pytest
import torch

from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.transformers.llm_inference_utils import (
    _get_default_task_for_llm_inference_task,
    _get_finish_reason,
    _get_output_and_usage_from_tensor,
    _get_stopping_criteria,
    _get_token_usage,
    convert_data_messages_with_chat_template,
    infer_signature_from_llm_inference_task,
    preprocess_llm_inference_params,
)
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    COMPLETIONS_MODEL_INPUT_SCHEMA,
    COMPLETIONS_MODEL_OUTPUT_SCHEMA,
)


def test_infer_signature_from_llm_inference_task():
    signature = infer_signature_from_llm_inference_task("llm/v1/completions")
    assert signature.inputs == COMPLETIONS_MODEL_INPUT_SCHEMA
    assert signature.outputs == COMPLETIONS_MODEL_OUTPUT_SCHEMA

    signature = infer_signature_from_llm_inference_task("llm/v1/chat")
    assert signature.inputs == CHAT_MODEL_INPUT_SCHEMA
    assert signature.outputs == CHAT_MODEL_OUTPUT_SCHEMA

    signature = infer_signature("hello", "world")
    with pytest.raises(MlflowException, match=r".*llm/v1/completions.*signature"):
        infer_signature_from_llm_inference_task("llm/v1/completions", signature)


class DummyTokenizer:
    def __call__(self, text: str, **kwargs):
        input_ids = list(map(int, text.split(" ")))
        return {"input_ids": torch.tensor([input_ids])}

    def decode(self, tensor, **kwargs):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.tolist()
        return " ".join([str(x) for x in tensor])

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [int(x) for x in tokens]

    def _tokenize(self, text: str):
        return [x for x in text.split(" ") if x]

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs):
        return " ".join(message["content"] for message in messages)


def test_apply_chat_template():
    tokenizer = DummyTokenizer()

    data1 = pd.DataFrame(
        {
            "messages": pd.Series(
                [[{"role": "A", "content": "one"}, {"role": "B", "content": "two"}]]
            ),
            "random": ["value"],
        }
    )

    # Test that the function modifies the data in place for Chat task
    convert_data_messages_with_chat_template(data1, tokenizer)

    expected_data = pd.DataFrame({"random": ["value"], "prompt": ["one two"]})
    pd.testing.assert_frame_equal(data1, expected_data)


def test_preprocess_llm_inference_params():
    data = pd.DataFrame(
        {
            "prompt": ["Hello world!"],
            "temperature": [0.7],
            "max_tokens": [100],
        }
    )

    data, params = preprocess_llm_inference_params(data, flavor_config=None)

    # Test that OpenAI params are separated from data and replaced with Hugging Face params
    assert data == ["Hello world!"]
    assert params == {"max_new_tokens": 100, "temperature": 0.7}


@mock.patch("transformers.AutoTokenizer.from_pretrained")
def test_stopping_criteria(mock_from_pretrained):
    mock_from_pretrained.return_value = DummyTokenizer()

    stopping_criteria = _get_stopping_criteria(stop=None, model_name=None)
    assert stopping_criteria is None

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    scores = torch.ones(1, 5)

    stopping_criteria = _get_stopping_criteria(stop="5", model_name="my/model")
    stopping_criteria_matches = [f(input_ids, scores) for f in stopping_criteria]
    assert stopping_criteria_matches == [True, True]

    stopping_criteria = _get_stopping_criteria(stop=["100", "5"], model_name="my/model")
    stopping_criteria_matches = [f(input_ids, scores) for f in stopping_criteria]
    assert stopping_criteria_matches == [False, False, True, True]


def test_output_dict_for_completions():
    prompt = "1 2 3"
    output_tensor = [1, 2, 3, 4, 5]
    flavor_config = {"source_model_name": "gpt2"}
    model_config = {"max_new_tokens": 2}
    inference_task = "llm/v1/completions"

    pipeline = mock.MagicMock()
    pipeline.tokenizer = DummyTokenizer()

    output_dict = _get_output_and_usage_from_tensor(
        prompt, output_tensor, pipeline, flavor_config, model_config, inference_task
    )

    # Test UUID validity
    uuid.UUID(output_dict["id"])

    assert output_dict["object"] == "text_completion"
    assert output_dict["model"] == "gpt2"

    assert output_dict["choices"][0]["text"] == "4 5"
    assert output_dict["choices"][0]["finish_reason"] == "length"

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


@pytest.mark.parametrize(
    ("inference_task", "expected_task"),
    [
        ("llm/v1/completions", "text-generation"),
        ("llm/v1/chat", "text-generation"),
        (None, None),
    ],
)
def test_default_task_for_llm_inference_task(inference_task, expected_task):
    assert _get_default_task_for_llm_inference_task(inference_task) == expected_task
