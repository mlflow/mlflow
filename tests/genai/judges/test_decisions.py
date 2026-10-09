import json
from types import SimpleNamespace
from typing import Literal
from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import ModelAPI, decisions, make_judge
from mlflow.genai.judges.typesafe import _GatewayEndpointNotSystemOne
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import AssessmentMetadataKey


def _decision_response(answer):
    return SimpleNamespace(
        answers=[answer],
        model="gpt-6-luna",
        usage=SimpleNamespace(input_tokens=12, output_tokens=0),
    )


@pytest.mark.parametrize(("probability", "expected"), [(0.49, False), (0.5, True), (1.0, True)])
def test_decisions_predicate_preserves_probability(monkeypatch, probability, expected):
    request = {}

    def create(**kwargs):
        request.update(kwargs)
        return _decision_response(
            SimpleNamespace(name="evaluation", type="predicate", probability=probability)
        )

    monkeypatch.setattr(
        decisions,
        "_openai_client",
        lambda *_: SimpleNamespace(decisions=SimpleNamespace(create=create)),
    )
    judge = make_judge(
        name="correctness",
        instructions="Is {{ outputs }} correct for {{ inputs }}?",
        model="openai:/gpt-6-luna",
        model_api=ModelAPI.DECISIONS,
        feedback_value_type=bool,
    )
    feedback = judge(inputs={"question": "2+2"}, outputs="4")

    assert request["model"] == "gpt-6-luna"
    assert request["questions"] == [
        {
            "name": "evaluation",
            "instructions": 'Is the "outputs" field in the input correct for the "inputs" field '
            "in the input?",
            "type": "predicate",
        }
    ]
    assert '"question": "2+2"' in request["input"]
    assert feedback.value is expected
    assert feedback.rationale is None
    assert json.loads(feedback.metadata["openai.decisions.probability"]) == probability
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_INPUT_TOKENS] == "12"


def test_decisions_choice_and_registered_judge_roundtrip(monkeypatch):
    request = {}

    def create(**kwargs):
        request.update(kwargs)
        return _decision_response(
            SimpleNamespace(
                name="evaluation",
                type="choice",
                choice="pass",
                confidence=0.8,
                probabilities=[
                    SimpleNamespace(value="pass", probability=0.8),
                    SimpleNamespace(value="fail", probability=0.2),
                ],
            )
        )

    monkeypatch.setattr(
        decisions,
        "_openai_client",
        lambda *_: SimpleNamespace(decisions=SimpleNamespace(create=create)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api=ModelAPI.DECISIONS,
        feedback_value_type=Literal["pass", "fail"],
    )
    restored = Scorer.model_validate(judge.model_dump())
    feedback = restored(outputs="good")

    assert restored.model_api is ModelAPI.DECISIONS
    assert restored.model_dump()["instructions_judge_pydantic_data"]["model_api"] == "decisions"
    assert request["questions"][0]["choices"] == [{"value": "pass"}, {"value": "fail"}]
    assert feedback.value == "pass"
    assert json.loads(feedback.metadata["openai.decisions.probabilities"]) == [
        {"value": "pass", "probability": 0.8},
        {"value": "fail", "probability": 0.2},
    ]


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (SimpleNamespace(name="evaluation", type="refusal"), "refused"),
        (
            SimpleNamespace(name="evaluation", type="predicate", probability=float("nan")),
            "invalid response",
        ),
    ],
)
def test_decisions_refusal_and_malformed_answers_fail(monkeypatch, answer, error):
    monkeypatch.setattr(
        decisions,
        "_openai_client",
        lambda *_: SimpleNamespace(
            decisions=SimpleNamespace(create=lambda **_: _decision_response(answer))
        ),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="decisions",
        feedback_value_type=bool,
    )
    with pytest.raises(MlflowException, match=error):
        judge(outputs="text")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"model_api": "unknown"}, "model_api must"),
        ({"model_api": ""}, "model_api must"),
        ({"model_api": "responses"}, "model_api must"),
        ({"model_api": "decisions", "feedback_value_type": str}, "require"),
        ({"model_api": "decisions", "feedback_value_type": Literal[1, 2]}, "string"),
        ({"model_api": "decisions", "inference_params": {"temperature": 0}}, "inference"),
        ({"model_api": "decisions", "generate_rationale_first": True}, "rationale"),
    ],
)
def test_decisions_reject_unsupported_options(kwargs, error):
    with pytest.raises(MlflowException, match=error):
        make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="openai:/gpt-6-luna",
            **{"feedback_value_type": bool, **kwargs},
        )


@pytest.mark.parametrize(
    ("model", "model_api"),
    [
        ("openai:/gpt-6-luna", ModelAPI.DECISIONS),
        ("typesafe:/jev-latest", ModelAPI.DEFAULT),
        ("gateway:/jev-endpoint", ModelAPI.DECISIONS),
    ],
)
def test_decisions_reject_trace_tools(model, model_api):
    with pytest.raises(MlflowException, match="trace.*tool calling"):
        make_judge(
            name="quality",
            instructions="Judge {{ trace }}.",
            model=model,
            model_api=model_api,
            feedback_value_type=bool,
        )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai:/gpt-6-luna", ModelAPI.CHAT_COMPLETIONS),
        ("typesafe:/jev-latest", ModelAPI.DECISIONS),
        ("gateway:/judge-endpoint", ModelAPI.DEFAULT),
        ("anthropic:/claude-sonnet", ModelAPI.CHAT_COMPLETIONS),
    ],
)
def test_default_api_uses_provider_route(model, expected):
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model=model,
        feedback_value_type=bool,
    )
    assert judge.model_api is ModelAPI.DEFAULT
    assert judge._resolved_model_api is expected
    assert "model_api" not in judge.model_dump()["instructions_judge_pydantic_data"]


@pytest.mark.parametrize(
    ("model", "model_api"),
    [
        ("typesafe:/jev-latest", ModelAPI.CHAT_COMPLETIONS),
        ("anthropic:/claude-sonnet", ModelAPI.DECISIONS),
    ],
)
def test_model_api_rejects_unsupported_provider_route(model, model_api):
    with pytest.raises(MlflowException, match="not supported"):
        make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model=model,
            model_api=model_api,
            feedback_value_type=bool,
        )


@pytest.mark.parametrize("model_api", [ModelAPI.DEFAULT, ModelAPI.DECISIONS])
def test_typesafe_default_and_explicit_decisions_use_native_route(model_api):
    result = Feedback(name="quality", value=True)
    with (
        mock.patch(
            "mlflow.genai.judges.instructions_judge._invoke_typesafe_judge",
            return_value=result,
        ) as invoke_decisions,
        mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as invoke_chat,
    ):
        judge = make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="typesafe:/jev-latest",
            model_api=model_api,
            feedback_value_type=bool,
        )
        assert judge(outputs="good") is result

    invoke_decisions.assert_called_once()
    invoke_chat.assert_not_called()


@pytest.mark.parametrize(
    ("model_api", "route"),
    [
        (ModelAPI.DEFAULT, "auto"),
        (ModelAPI.CHAT_COMPLETIONS, "chat"),
        (ModelAPI.DECISIONS, "decision"),
    ],
)
def test_gateway_model_api_controls_route(model_api, route):
    result = Feedback(name="quality", value=True)
    with (
        mock.patch(
            "mlflow.genai.judges.instructions_judge._invoke_gateway_judge",
            return_value=result,
        ) as invoke_auto,
        mock.patch(
            "mlflow.genai.judges.instructions_judge._invoke_typesafe_judge",
            return_value=result,
        ) as invoke_decisions,
        mock.patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=result,
        ) as invoke_chat,
    ):
        judge = make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="gateway:/judge-endpoint",
            model_api=model_api,
            feedback_value_type=bool,
        )
        assert judge(outputs="good") is result

    assert invoke_auto.call_count == (route == "auto")
    assert invoke_chat.call_count == (route == "chat")
    assert invoke_decisions.call_count == (route == "decision")


def test_explicit_decisions_reject_chat_gateway_endpoint():
    with mock.patch(
        "mlflow.genai.judges.instructions_judge._invoke_typesafe_judge",
        side_effect=_GatewayEndpointNotSystemOne,
    ):
        judge = make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="gateway:/chat-endpoint",
            model_api=ModelAPI.DECISIONS,
            feedback_value_type=bool,
        )
        with pytest.raises(MlflowException, match="does not serve a System One model"):
            judge(outputs="good")
