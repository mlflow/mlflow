import json
from typing import Literal
from unittest import mock

import pytest
import requests

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import ModelAPI, make_judge
from mlflow.genai.judges.model_api import _select_model_api_mode
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.utils.rest_utils import MlflowHostCreds

_MODEL = "databricks:/system.ai.typesafeai-jev-1-14-0"
_HTTP_REQUEST = "mlflow.genai.judges.typesafe.http_request"
_HOST_CREDS = "mlflow.genai.judges.typesafe.get_databricks_host_creds"


def _response(answer=None, status_code=200):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps({
        "model": "jev-1.14.0",
        "answers": {"evaluation": answer or {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 123, "output_tokens": 0},
    }).encode()
    return response


@pytest.mark.parametrize(
    ("feedback_value_type", "answer", "expected"),
    [
        (bool, {"type": "noul", "noul": 0.8}, True),
        (
            Literal["pass", "fail"],
            {
                "type": "choice",
                "choice": "pass",
                "probabilities": {"pass": 0.8, "fail": 0.2},
                "confidence": 0.6,
            },
            "pass",
        ),
    ],
)
def test_databricks_judge_serialization_and_execution(feedback_value_type, answer, expected):
    judge = make_judge(
        name="quality",
        instructions="Does {{ outputs }} correctly answer {{ inputs }}?",
        model=_MODEL,
        model_api="decisions",
        feedback_value_type=feedback_value_type,
    )
    serialized = judge.model_dump()
    data = serialized["instructions_judge_pydantic_data"]
    assert data["model"] == _MODEL
    assert data["model_api"] == "decisions"
    restored = Scorer.model_validate(json.loads(json.dumps(serialized)))
    creds = MlflowHostCreds("https://workspace.databricks.com", token="workspace-token")

    with (
        mock.patch(_HOST_CREDS, return_value=creds),
        mock.patch(_HTTP_REQUEST, return_value=_response(answer)) as request,
        mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as chat,
    ):
        feedback = restored.run(inputs={"question": "2+2"}, outputs="4")

    chat.assert_not_called()
    request.assert_called_once()
    call = request.call_args.kwargs
    assert call["host_creds"] is creds
    assert call["endpoint"] == "/ai-gateway/typesafe/v1/systemone"
    assert call["method"] == "POST"
    assert call["json"]["model"] == "system.ai.typesafeai-jev-1-14-0"
    assert call["json"]["state"] == {"inputs": {"question": "2+2"}, "outputs": "4"}
    question = call["json"]["questions"]["evaluation"]
    assert question["type"] == answer["type"]
    assert question["instructions"] == "Does state.outputs correctly answer state.inputs?"
    assert feedback.value == expected
    assert feedback.rationale is None
    assert feedback.source.source_id == _MODEL
    assert feedback.metadata["typesafe.model"] == "jev-1.14.0"
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_INPUT_TOKENS] == "123"
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS] == "0"
    assert "workspace-token" not in json.dumps(restored.model_dump())


@pytest.mark.parametrize(
    ("tracking_uri", "profile_uri"),
    [
        ("databricks", "databricks"),
        ("databricks://evaluation", "databricks://evaluation"),
        ("http://localhost:5000", None),
        ("sqlite:///mlflow.db", None),
    ],
)
def test_databricks_auth_is_resolved_at_invocation(monkeypatch, tracking_uri, profile_uri):
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused-typesafe-key")
    monkeypatch.setenv("MLFLOW_GATEWAY_URI", "http://unused-oss-gateway:5000")
    with mock.patch(_HOST_CREDS) as get_creds:
        judge = make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="databricks:/catalog.schema.my-evaluator",
            model_api=ModelAPI.DECISIONS,
            feedback_value_type=bool,
        )
        restored = Scorer.model_validate(judge.model_dump())
    get_creds.assert_not_called()

    first_creds = MlflowHostCreds("https://workspace.databricks.com", token="first-token")
    second_creds = MlflowHostCreds("https://workspace.databricks.com", token="second-token")
    with (
        mock.patch("mlflow.genai.judges.typesafe.get_tracking_uri", return_value=tracking_uri),
        mock.patch(_HOST_CREDS, side_effect=[first_creds, second_creds]) as get_creds,
        mock.patch(_HTTP_REQUEST, return_value=_response()) as request,
    ):
        restored(outputs="first")
        restored(outputs="second")

    assert get_creds.call_args_list == [mock.call(profile_uri), mock.call(profile_uri)]
    assert [call.kwargs["host_creds"] for call in request.call_args_list] == [
        first_creds,
        second_creds,
    ]
    assert request.call_args.kwargs["json"]["model"] == "catalog.schema.my-evaluator"


@pytest.mark.parametrize("model_api", [ModelAPI.DEFAULT, ModelAPI.CHAT_COMPLETIONS])
def test_databricks_chat_routing_is_unchanged(model_api):
    result = Feedback(name="quality", value=True)
    with (
        mock.patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model", return_value=result
        ) as chat,
        mock.patch(_HTTP_REQUEST) as request,
    ):
        judge = make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="databricks:/system.ai.claude-haiku-4-5",
            model_api=model_api,
            feedback_value_type=bool,
        )
        assert judge(outputs="good") is result
    chat.assert_called_once()
    request.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"base_url": "https://example.com"}, "base_url"),
        ({"extra_headers": {"Authorization": "Bearer token"}}, "extra_headers"),
        ({"inference_params": {"temperature": 0}}, "inference_params"),
        ({"feedback_value_type": str}, "bool or finite Literal"),
        ({"model": "databricks:/databricks-typesafeai-jev-1-14-0"}, "Unity Catalog"),
        ({"instructions": "Judge {{ trace }}."}, "trace.*tool calling"),
        ({"generate_rationale_first": True}, "rationale"),
    ],
)
def test_databricks_unsupported_options_fail_before_network(kwargs, error):
    with (
        mock.patch(_HOST_CREDS) as get_creds,
        mock.patch(_HTTP_REQUEST) as request,
        pytest.raises(MlflowException, match=error),
    ):
        make_judge(**{
            "name": "quality",
            "instructions": "Judge {{ outputs }}.",
            "model": _MODEL,
            "model_api": ModelAPI.DECISIONS,
            "feedback_value_type": bool,
            **kwargs,
        })(outputs="good")
    get_creds.assert_not_called()
    request.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429, 500])
def test_databricks_errors_do_not_fall_back_to_chat(status_code):
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model=_MODEL,
        model_api=ModelAPI.DECISIONS,
        feedback_value_type=bool,
    )
    with (
        mock.patch(_HOST_CREDS),
        mock.patch(_HTTP_REQUEST, return_value=_response(status_code=status_code)),
        mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as chat,
        pytest.raises(MlflowException, match=f"HTTP {status_code}"),
    ):
        judge(outputs="good")
    chat.assert_not_called()


def test_default_databricks_managed_judge_does_not_support_decisions():
    with pytest.raises(MlflowException, match="default Databricks judge"):
        _select_model_api_mode("databricks", ModelAPI.DECISIONS)
