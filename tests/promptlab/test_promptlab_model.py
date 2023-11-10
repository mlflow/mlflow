from unittest import mock

import pandas as pd

from mlflow._promptlab import _PromptlabModel
from mlflow.entities.param import Param


def construct_model(route):
    prompt_parameters = [Param(key="thing", value="books")]
    model_parameters = [Param(key="temperature", value=0.5), Param(key="max_tokens", value=10)]
    prompt_template = "Write me a story about {{ thing }}."

    return _PromptlabModel(prompt_template, prompt_parameters, model_parameters, route)


def test_promptlab_prompt_replacement():
    data = pd.DataFrame(
        data=[
            {"thing": "books"},
            {"thing": "coffee"},
            {"thing": "nothing"},
        ]
    )

    model = construct_model("completions")
    with mock.patch("mlflow.gateway.query") as mock_query:
        model.predict(data)

        calls = [
            mock.call(
                route="completions",
                data={
                    "prompt": f"Write me a story about {thing}.",
                    "temperature": 0.5,
                    "max_tokens": 10,
                },
            )
            for thing in data["thing"]
        ]

        mock_query.assert_has_calls(calls, any_order=True)


def test_promptlab_works_with_chat_route():
    mock_response = {
        "candidates": [
            {"message": {"role": "user", "content": "test"}, "metadata": {"finish_reason": "stop"}}
        ]
    }
    model = construct_model("chat")

    with mock.patch("mlflow.gateway.query", return_value=mock_response):
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))

        assert response == ["test"]


def test_promptlab_works_with_completions_route():
    mock_response = {
        "candidates": [
            {
                "text": "test",
                "metadata": {"finish_reason": "stop"},
            }
        ]
    }
    model = construct_model("completions")

    with mock.patch("mlflow.gateway.query", return_value=mock_response):
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))

        assert response == ["test"]
