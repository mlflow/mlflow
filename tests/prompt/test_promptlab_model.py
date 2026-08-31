from unittest import mock

import pandas as pd

from mlflow.deployments import set_deployments_target
from mlflow.entities.param import Param
from mlflow.prompt.promptlab_model import _load_pyfunc, _PromptlabModel, save_model

set_deployments_target("http://localhost:5000")


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
        "mlflow.deployments.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/completions"),
    )

    with (
        get_route_patch,
        mock.patch("mlflow.deployments.MlflowDeploymentClient.predict") as mock_query,
    ):
        model.predict(data)

        calls = [
            mock.call(
                endpoint="completions",
                inputs={
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
        "mlflow.deployments.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/chat"),
    )

    with (
        get_route_patch,
        mock.patch("mlflow.deployments.MlflowDeploymentClient.predict", return_value=mock_response),
    ):
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
        "mlflow.deployments.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/completions"),
    )

    with (
        get_route_patch,
        mock.patch("mlflow.deployments.MlflowDeploymentClient.predict", return_value=mock_response),
    ):
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))

        assert response == ["test"]


def test_save_model_preserves_non_ascii_text(tmp_path):
    prompt_template = "{{ object }}は何色ですか"
    model_path = tmp_path / "model"

    save_model(
        path=str(model_path),
        prompt_template=prompt_template,
        prompt_parameters=[Param(key="object", value="空")],
        model_parameters=[Param(key="temperature", value=0.5)],
        model_route="completions",
        pip_requirements=["mlflow"],
    )

    written = (model_path / "parameters.yaml").read_text(encoding="utf-8")
    assert prompt_template in written
    assert r"\u" not in written

    loaded = _load_pyfunc(str(model_path))
    assert loaded.prompt_template == prompt_template
    assert [(p.key, p.value) for p in loaded.prompt_parameters] == [("object", "空")]
