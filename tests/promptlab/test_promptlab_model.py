from unittest import mock

import pandas as pd

from mlflow.entities.param import Param
from mlflow.gateway import set_gateway_uri
from mlflow.promptlab import _PromptlabModel

set_gateway_uri("http://localhost:5000")


def construct_model(route):
    return _PromptlabModel(
        "Write me a story about {{ thing }}.",
        [Param(key="thing", value="books")],
        [Param(key="temperature", value=0.5), Param(key="max_tokens", value=10)],
        route,
    )


def test_promptlab_prompt_replacement():
    data = pd.DataFrame(
        data=[
            {"thing": "books"},
            {"thing": "coffee"},
            {"thing": "nothing"},
        ]
    )

    model = construct_model("completions")
    get_route_patch = mock.patch(
        "mlflow.gateway.get_route", return_value=mock.Mock(route_type="llm/v1/completions")
    )

    with get_route_patch, mock.patch("mlflow.gateway.query") as mock_query:
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
        "choices": [
            {"message": {"role": "user", "content": "test"}, "metadata": {"finish_reason": "stop"}}
        ]
    }
    model = construct_model("chat")
    get_route_patch = mock.patch(
        "mlflow.gateway.get_route",
        return_value=mock.Mock(route_type="llm/v1/chat"),
    )

    with get_route_patch, mock.patch("mlflow.gateway.query", return_value=mock_response):
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))

        assert response == ["test"]


def test_promptlab_works_with_completions_route():
    mock_response = {
        "choices": [
            {
                "text": "test",
                "metadata": {"finish_reason": "stop"},
            }
        ]
    }
    model = construct_model("completions")
    get_route_patch = mock.patch(
        "mlflow.gateway.get_route", return_value=mock.Mock(route_type="llm/v1/completions")
    )

    with get_route_patch, mock.patch("mlflow.gateway.query", return_value=mock_response):
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))

        assert response == ["test"]
