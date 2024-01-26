from mlflow.transformers.inference_utils import (
    _get_finish_reason,
    _get_output_and_usage_from_tensor,
    _get_token_usage,
)


def test_output_dict_for_completions(text_generation_pipeline):
    prompt = "How to learn Python in 3 weeks?"
    output_tensor = text_generation_pipeline.tokenizer("How to learn Python in 3 weeks? First,")[
        "input_ids"
    ]

    model_config = {"max_new_tokens": 10}
    inference_task = "llm/v1/completions"

    output_dict = _get_output_and_usage_from_tensor(
        prompt, output_tensor, text_generation_pipeline, model_config, inference_task
    )

    assert output_dict["text"] == "First,"
    assert output_dict["finish_reason"] == "stop"

    usage = output_dict["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_token_usage(text_generation_pipeline):
    prompt = "."
    output_tensor = text_generation_pipeline.tokenizer(". So")["input_ids"]

    usage = _get_token_usage(prompt, output_tensor, text_generation_pipeline, {})
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == 2


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
