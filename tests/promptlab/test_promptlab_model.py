from unittest import mock

import pandas as pd

from mlflow._promptlab import _PromptlabModel
from mlflow.entities.param import Param


def test_promptlab_prompt_replacement():
    data = pd.DataFrame(
        data=[
            {"thing": "books"},
            {"thing": "coffee"},
            {"thing": "nothing"},
        ]
    )

    prompt_parameters = [Param(key="thing", value="books")]
    model_parameters = [Param(key="temperature", value=0.5), Param(key="max_tokens", value=10)]
    prompt_template = "Write me a story about {{ thing }}."
    model_route = "completions"

    model = _PromptlabModel(prompt_template, prompt_parameters, model_parameters, model_route)
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
