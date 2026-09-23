import json
from unittest import mock

import pytest
import requests
from pydantic import ValidationError

import mlflow
from mlflow.entities import (
    AssessmentSourceType,
    GatewayEndpointModelConfig,
    GatewayModelLinkageType,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import JevScorer, Scorer, ScorerSamplingConfig, make_jev_scorer
from mlflow.genai.scorers.base import ScorerKind
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.registry import get_scorer, list_scorers
from mlflow.genai.scorers.scorer_utils import (
    extract_model_from_serialized_scorer,
    update_model_in_serialized_scorer,
)
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


def _scorer(**kwargs):
    return make_jev_scorer(**{
        "name": "relevance",
        "model": "typesafe:/jev-latest",
        "question": "Does outputs address the question in inputs?",
        **kwargs,
    })


def _response(answer):
    return {
        "model": "jev-1.13.0",
        "answers": {"evaluation": answer},
        "usage": {"input_tokens": 30, "output_tokens": 2},
    }


@pytest.fixture
def direct_request(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    with mock.patch("mlflow.genai.scorers.jev._get_http_response_with_retries") as request:
        request.return_value.status_code = 200
        request.return_value.json.return_value = _response({"type": "noul", "noul": 0.8})
        yield request


@pytest.mark.parametrize(
    ("threshold", "expected"), [(None, 0.8), (0.7, True), (0.8, True), (0.9, False)]
)
def test_noul_feedback_and_request(direct_request, threshold, expected):
    scorer = _scorer(threshold=threshold, criteria={"true": "Relevant", "false": "Irrelevant"})
    feedback = scorer(inputs={"question": "2 + 2?"}, outputs="4", expectations={"answer": "4"})

    assert feedback.name == "relevance"
    assert feedback.value == expected
    assert type(feedback.value) is type(expected)
    assert feedback.rationale is None
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == scorer.model
    assert feedback.metadata["jev.probability"] == "0.8"
    assert feedback.metadata["jev.model"] == "jev-1.13.0"
    assert "jev.confidence" not in feedback.metadata
    call = direct_request.call_args.kwargs
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"] == {"Authorization": "Bearer test-typesafe-key"}
    assert call["allow_redirects"] is False
    assert 429 in call["retry_codes"]
    assert 529 in call["retry_codes"]
    assert call["json"] == {
        "model": "jev-latest",
        "state": {
            "inputs": {"question": "2 + 2?"},
            "outputs": "4",
            "expectations": {"answer": "4"},
        },
        "questions": {
            "evaluation": {
                "type": "noul",
                "instructions": scorer.question,
                "criteria": {"true": "Relevant", "false": "Irrelevant"},
            }
        },
    }


@pytest.mark.parametrize(
    ("answer_type", "criteria", "answer", "expected"),
    [
        (
            "choice",
            {"billing": "Payments", "technical": "Bugs"},
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1},
                "confidence": 0.8,
            },
            "billing",
        ),
        (
            "score",
            ["Poor", "Fair", "Good"],
            {
                "type": "score",
                "score": 1.7,
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                "confidence": 0.7,
                "legend": {"0": "Poor", "1": "Fair", "2": "Good"},
            },
            1.7,
        ),
    ],
)
def test_typed_feedback(direct_request, answer_type, criteria, answer, expected):
    direct_request.return_value.json.return_value = _response(answer)
    feedback = _scorer(answer_type=answer_type, criteria=criteria)(outputs="Some response")

    assert feedback.value == expected
    assert feedback.rationale is None
    assert json.loads(feedback.metadata["jev.probabilities"]) == answer["probabilities"]
    assert json.loads(feedback.metadata["jev.confidence"]) == answer["confidence"]
    if answer_type == "score":
        assert json.loads(feedback.metadata["jev.legend"]) == answer["legend"]
    assert all(isinstance(value, str) for value in feedback.metadata.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " "},
        {"question": ""},
        {"model": "openai:/gpt-4"},
        {"model": "typesafe:/"},
        {"model": "typesafe:/https://example.com"},
        {"model": "gateway:/endpoint?api_key=secret"},
        {"answer_type": "boolean"},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"criteria": {"yes": "Yes"}},
        {"criteria": ["Yes", "No"]},
        {"criteria": {"true": 1}},
        {"answer_type": "choice"},
        {"answer_type": "choice", "criteria": {}},
        {"answer_type": "choice", "criteria": {"": "Empty"}},
        {"answer_type": "choice", "criteria": {str(i): "Option" for i in range(256)}},
        {"answer_type": "choice", "criteria": {"a": "A"}, "threshold": 0.5},
        {"answer_type": "score", "criteria": ["Single"]},
        {"answer_type": "score", "criteria": ["Level"] * 11},
        {"answer_type": "score", "criteria": {"a": "A", "b": "B"}},
    ],
)
def test_invalid_configuration(kwargs, direct_request):
    with pytest.raises(ValidationError, match="validation error"):
        _scorer(**kwargs)
    direct_request.assert_not_called()


def test_serialization_preserves_configuration_without_credentials(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "do-not-serialize-this")
    scorer = JevScorer(
        name="rating",
        model="gateway:/evaluator",
        question="Rate outputs",
        answer_type="score",
        criteria=["Poor", "Good"],
        description="Quality rating",
        aggregations=["mean"],
        timeout=42,
    )
    serialized = scorer.model_dump()
    assert serialized["call_source"] is None
    assert "do-not-serialize-this" not in json.dumps(serialized)
    restored = Scorer.model_validate_json(json.dumps(serialized))
    assert isinstance(restored, JevScorer)
    assert restored.kind == ScorerKind.JEV
    assert restored.model_dump() == serialized
    assert restored.timeout == 42

    serialized["jev_scorer_pydantic_data"]["api_key"] = "never-echo-me"
    with pytest.raises(MlflowException, match="Extra inputs are not permitted") as exc:
        Scorer.model_validate(serialized)
    assert "never-echo-me" not in str(exc.value)


def test_serialization_rejects_multiple_variants():
    serialized = _scorer().model_dump()
    serialized["call_source"] = "return True"
    with pytest.raises(MlflowException, match="multiple types"):
        Scorer.model_validate(serialized)


def test_gateway_model_rewrite():
    serialized = _scorer(model="gateway:/evaluator").model_dump()
    updated = update_model_in_serialized_scorer(serialized, "gateway:/endpoint-id")
    assert extract_model_from_serialized_scorer(updated) == "gateway:/endpoint-id"
    assert extract_model_from_serialized_scorer(serialized) == "gateway:/evaluator"
    assert Scorer.model_validate(updated).question == _scorer().question


@pytest.mark.parametrize("endpoint", ["_jev", "-jev", ".jev"])
def test_gateway_accepts_valid_endpoint_names(endpoint):
    scorer = _scorer(model=f"gateway:/{endpoint}")
    assert Scorer.model_validate(scorer.model_dump()).model == f"gateway:/{endpoint}"


@pytest.mark.parametrize("status", [302, 401, 403, 422, 429, 529])
def test_http_failure_does_not_expose_response(status, direct_request):
    direct_request.return_value.status_code = status
    direct_request.return_value.text = "provider error includes test-typesafe-key"
    with pytest.raises(MlflowException, match=f"HTTP {status}") as exc:
        _scorer()(outputs="test")
    assert "test-typesafe-key" not in str(exc.value)


def test_connection_failure_is_sanitized(direct_request):
    direct_request.side_effect = requests.ConnectionError("test-typesafe-key")
    with pytest.raises(MlflowException, match="Failed to connect") as exc:
        _scorer()(outputs="test")
    assert "test-typesafe-key" not in str(exc.value)


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with mock.patch("mlflow.genai.scorers.jev._get_http_response_with_retries") as request:
        with pytest.raises(MlflowException, match="Set TYPESAFE_API_KEY"):
            _scorer()(outputs="test")
    request.assert_not_called()


def test_invalid_json(direct_request):
    direct_request.return_value.json.side_effect = ValueError("private response content")
    with pytest.raises(MlflowException, match="invalid JSON") as exc:
        _scorer()(outputs="test")
    assert "private response content" not in str(exc.value)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"answers": {"other": {"type": "noul", "noul": 0.5}}},
        _response({"type": "choice", "noul": 0.5}),
        _response({"type": "noul", "noul": "0.5"}),
        _response({"type": "noul", "noul": True}),
        _response({"type": "noul", "noul": -0.1}),
        _response({"type": "noul", "noul": float("nan")}),
        _response({"type": "noul", "noul": float("inf")}),
    ],
)
def test_invalid_noul_response(response, direct_request):
    direct_request.return_value.json.return_value = response
    with pytest.raises(MlflowException, match="invalid noul answer"):
        _scorer()(outputs="test")


@pytest.mark.parametrize(
    "changes",
    [
        {"choice": "unknown"},
        {"probabilities": {"a": 0.9}},
        {"probabilities": {"a": 0.9, "b": 0.9}},
        {"probabilities": {"a": True, "b": 0}},
        {"confidence": -0.1},
        {"confidence": "high"},
    ],
)
def test_invalid_choice_response(changes, direct_request):
    answer = {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.9, "b": 0.1},
        "confidence": 0.8,
        **changes,
    }
    direct_request.return_value.json.return_value = _response(answer)
    with pytest.raises(MlflowException, match="invalid choice answer"):
        _scorer(answer_type="choice", criteria={"a": "A", "b": "B"})(outputs="test")


@pytest.mark.parametrize("auth", ["basic", "token"])
def test_gateway_uses_tracking_auth_and_workspace(monkeypatch, auth):
    monkeypatch.setenv("MLFLOW_GATEWAY_URI", "https://mlflow.example.com/prefix")
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-forward")
    if auth == "basic":
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "password")
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "tracking-token")
    with (
        WorkspaceContext("team-a"),
        mock.patch("mlflow.utils.rest_utils._get_http_response_with_retries") as request,
    ):
        request.return_value.status_code = 200
        request.return_value.json.return_value = _response({"type": "noul", "noul": 0.8})
        assert _scorer(model="gateway:/evaluator")(outputs="test").value == 0.8

    call = request.call_args.kwargs
    assert request.call_args.args[1] == (
        "https://mlflow.example.com/prefix/gateway/typesafe/v1/systemone"
    )
    assert call["json"]["model"] == "evaluator"
    assert call["headers"][WORKSPACE_HEADER_NAME] == "team-a"
    assert call["headers"]["Authorization"] == (
        "Basic YWxpY2U6cGFzc3dvcmQ=" if auth == "basic" else "Bearer tracking-token"
    )
    assert "must-not-forward" not in repr(call)
    assert call["allow_redirects"] is False


def test_direct_request_does_not_forward_tracking_auth_or_workspace(direct_request, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "private-tracking-token")
    with WorkspaceContext("private-workspace"):
        _scorer()(outputs="test")
    assert direct_request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test-typesafe-key"
    }


def test_evaluate_and_trace_input_mapping(direct_request):
    with mlflow.start_span(name="application") as span:
        span.set_inputs({"question": "What is 2 + 2?"})
        span.set_outputs("4")
    trace = mlflow.get_trace(span.trace_id)
    mlflow.log_expectation(trace_id=trace.info.trace_id, name="answer", value="4")
    trace = mlflow.get_trace(span.trace_id)
    feedback = _scorer()(trace=trace)
    assert feedback.trace_id == trace.info.trace_id
    assert direct_request.call_args.kwargs["json"]["state"] == {
        "inputs": {"question": "What is 2 + 2?"},
        "outputs": "4",
        "expectations": {"answer": "4"},
    }

    result = mlflow.genai.evaluate(
        data=[{"inputs": {"question": "2 + 2?"}, "outputs": "4"}],
        scorers=[_scorer()],
    )
    assert result.metrics["relevance/mean"] == 0.8


def test_registration_versions_and_online_sampling(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase")
    experiment_id = mlflow.create_experiment("jev-registration")
    store = _get_store()
    secret = store.create_gateway_secret(
        secret_name="typesafe-key", secret_value={"api_key": "test-key"}, provider="typesafe"
    )
    model = store.create_gateway_model_definition(
        name="jev-model",
        secret_id=secret.secret_id,
        provider="typesafe",
        model_name="jev-latest",
    )
    endpoint = store.create_gateway_endpoint(
        name="evaluator",
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            )
        ],
    )
    scorer = _scorer(model="gateway:/evaluator")
    first = scorer.register(experiment_id=experiment_id)
    assert first.scorer_version == 1
    scorer.threshold = 0.7
    second = scorer.register(experiment_id=experiment_id)
    assert second.scorer_version == 2
    assert get_scorer(name=scorer.name, experiment_id=experiment_id, version=1).threshold is None
    assert list_scorers(experiment_id=experiment_id)[0].threshold == 0.7

    started = second.start(sampling_config=ScorerSamplingConfig(sample_rate=1))
    assert started.sample_rate == 1
    active = store.get_active_online_scorers()
    assert len(active) == 1
    sampler = OnlineScorerSampler(active)
    loaded = sampler.group_scorers_by_filter(session_level=False)[None][0]
    assert isinstance(loaded, JevScorer)
    assert loaded.scorer_version == 2
    assert loaded.model == "gateway:/evaluator"

    store.update_gateway_endpoint(endpoint.endpoint_id, name="renamed-evaluator")
    assert get_scorer(name=scorer.name, experiment_id=experiment_id).model == (
        "gateway:/renamed-evaluator"
    )
    assert started.stop().sample_rate == 0
    assert store.get_active_online_scorers() == []

    store.delete_gateway_endpoint(endpoint.endpoint_id)
    restored = get_scorer(name=scorer.name, experiment_id=experiment_id)
    assert restored.model is None
    assert list_scorers(experiment_id=experiment_id)[0].model is None
    with mock.patch("mlflow.genai.scorers.jev.JevScorer._invoke") as invoke:
        with pytest.raises(MlflowException, match="gateway endpoint may have been deleted"):
            restored(outputs="test")
        with pytest.raises(MlflowException, match="requires a gateway:/ endpoint"):
            restored.register(experiment_id=experiment_id)
    invoke.assert_not_called()


def test_factory_requires_model():
    with pytest.raises(MlflowException, match="A model is required"):
        _scorer(model=None)


def test_registration_rejects_direct_model(direct_request):
    with pytest.raises(MlflowException, match="requires a gateway:/ endpoint"):
        _scorer().register()
    direct_request.assert_not_called()


def test_registration_rejects_databricks():
    with mock.patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        with pytest.raises(MlflowException, match="OSS MLflow tracking server"):
            _scorer(model="gateway:/evaluator").register()
