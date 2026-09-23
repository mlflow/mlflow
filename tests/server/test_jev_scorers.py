import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities import ScorerVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import make_jev_scorer
from mlflow.genai.scorers.job import invoke_scorer_job
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.protos.service_pb2 import RegisterScorer
from mlflow.server import app
from mlflow.server.handlers import (
    _invoke_scorer_handler,
    _register_scorer,
    _validate_serialized_scorer_payload,
)
from mlflow.utils.workspace_context import WorkspaceContext


def _serialized_scorer(model="gateway:/evaluator"):
    return make_jev_scorer(
        name="relevance", model=model, question="Is outputs relevant to inputs?", threshold=0.7
    ).model_dump()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("model", ["typesafe:/jev-latest", "gateway:/evaluator"])
def test_server_validates_jev_configuration_without_invoking(nested, model):
    serialized = _serialized_scorer(model)
    if nested:
        serialized = {
            "name": "ensemble",
            "ensemble_scorer_data": {"ensemble_fn": "mean", "scorers": [serialized]},
        }
    with mock.patch("mlflow.genai.scorers.jev.JevScorer._invoke") as invoke:
        if model.startswith("typesafe:/"):
            with pytest.raises(MlflowException, match="Server-side Jev scorers require a gateway"):
                _validate_serialized_scorer_payload(json.dumps(serialized))
        else:
            _validate_serialized_scorer_payload(json.dumps(serialized))
    invoke.assert_not_called()


@pytest.mark.parametrize("field", ["api_key", "base_url", "extra_headers"])
def test_server_rejects_jev_credentials_and_destination_options(field):
    serialized = _serialized_scorer()
    serialized["jev_scorer_pydantic_data"][field] = "private-value"
    with pytest.raises(MlflowException, match="Extra inputs are not permitted") as exc:
        _validate_serialized_scorer_payload(json.dumps(serialized))
    assert "private-value" not in str(exc.value)


def test_server_accepts_choice_labels_that_resemble_serialization_fields():
    scorer = make_jev_scorer(
        name="category",
        model="gateway:/evaluator",
        question="Choose the category",
        answer_type="choice",
        criteria={"jev_scorer_pydantic_data": "A category", "other": "Other"},
    )
    _validate_serialized_scorer_payload(json.dumps(scorer.model_dump()))


def test_server_rejects_missing_gateway_endpoint():
    serialized = _serialized_scorer()
    serialized["jev_scorer_pydantic_data"]["model"] = None
    with pytest.raises(MlflowException, match="require a gateway:/ endpoint"):
        _validate_serialized_scorer_payload(json.dumps(serialized))


def test_register_jev_scorer_handler_preserves_workspace():
    serialized = json.dumps(_serialized_scorer())
    request = RegisterScorer(experiment_id="123", name="relevance", serialized_scorer=serialized)
    with (
        WorkspaceContext("team-a"),
        mock.patch("mlflow.server.handlers._get_request_message", return_value=request),
        mock.patch("mlflow.server.handlers._get_tracking_store") as store,
    ):
        store.return_value.register_scorer.return_value = ScorerVersion(
            experiment_id="123",
            scorer_name="relevance",
            scorer_version=1,
            serialized_scorer=serialized,
            creation_time=0,
            scorer_id="scorer-id",
        )
        response = _register_scorer()
    assert response.status_code == 200
    store.return_value.register_scorer.assert_called_once_with("123", "relevance", serialized)
    assert response.get_json()["version"] == 1


@pytest.mark.parametrize("operation", ["register", "invoke"])
def test_server_handlers_reject_direct_jev_before_store_access(operation):
    serialized = json.dumps(_serialized_scorer("typesafe:/jev-latest"))
    with (
        app.test_request_context(
            method="POST",
            json={
                "experiment_id": "123",
                "serialized_scorer": serialized,
                "trace_ids": ["trace-1"],
            },
        ),
        mock.patch(
            "mlflow.server.handlers._get_request_message",
            return_value=RegisterScorer(
                experiment_id="123", name="relevance", serialized_scorer=serialized
            ),
        ),
        mock.patch("mlflow.server.handlers._get_tracking_store") as store,
        mock.patch("mlflow.genai.scorers.jev.JevScorer._invoke") as invoke,
    ):
        response = _register_scorer() if operation == "register" else _invoke_scorer_handler()
    assert response.status_code == 400
    assert "require a gateway:/ endpoint" in response.get_json()["message"]
    store.assert_not_called()
    invoke.assert_not_called()


def test_online_sampler_skips_stored_direct_jev(caplog):
    online_scorer = mock.Mock(
        serialized_scorer=json.dumps(_serialized_scorer("typesafe:/jev-latest"))
    )
    online_scorer.name = "relevance"
    with mock.patch("mlflow.genai.scorers.jev.JevScorer._invoke") as invoke:
        sampler = OnlineScorerSampler([online_scorer])
    assert sampler.group_scorers_by_filter(session_level=False) == {}
    assert "require a gateway:/ endpoint" in caplog.text
    invoke.assert_not_called()


def test_async_job_rejects_stored_direct_jev():
    with mock.patch("mlflow.genai.scorers.jev.JevScorer._invoke") as invoke:
        with pytest.raises(MlflowException, match="require a gateway:/ endpoint"):
            invoke_scorer_job(
                experiment_id="123",
                serialized_scorer=json.dumps(_serialized_scorer("typesafe:/jev-latest")),
                trace_ids=["trace-1"],
            )
    invoke.assert_not_called()


def test_async_job_evaluates_native_jev_configuration():
    with mlflow.start_span(name="application") as span:
        span.set_inputs({"question": "What is 2 + 2?"})
        span.set_outputs("4")
    trace = mlflow.get_trace(span.trace_id)
    with (
        mock.patch("mlflow.genai.scorers.job._get_tracking_store"),
        mock.patch(
            "mlflow.genai.scorers.job._fetch_traces_batch", return_value={span.trace_id: trace}
        ),
        mock.patch("mlflow.genai.scorers.jev.http_request") as request,
        mock.patch(
            "mlflow.genai.scorers.jev._resolve_gateway_uri",
            return_value="https://mlflow.example.com",
        ),
    ):
        request.return_value.status_code = 200
        request.return_value.json.return_value = {
            "model": "jev-1.13.0",
            "answers": {"evaluation": {"type": "noul", "noul": 0.8}},
        }
        results = invoke_scorer_job(
            experiment_id=trace.info.experiment_id,
            serialized_scorer=json.dumps(_serialized_scorer()),
            trace_ids=[span.trace_id],
            log_assessments=False,
            scorer_version=3,
        )
    assert results[span.trace_id]["failures"] == []
    assessment = results[span.trace_id]["assessments"][0]
    assert assessment["feedback"]["value"] is True
    assert assessment["metadata"]["jev.probability"] == "0.8"
    assert request.call_args.kwargs["json"]["state"]["outputs"] == "4"
