import json
from types import SimpleNamespace
from typing import Literal

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge, openai_api
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
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(decisions=SimpleNamespace(create=create)),
    )
    judge = make_judge(
        name="correctness",
        instructions="Is {{ outputs }} correct for {{ inputs }}?",
        model="openai:/gpt-6-luna",
        model_api="decisions",
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
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(decisions=SimpleNamespace(create=create)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="decisions",
        feedback_value_type=Literal["pass", "fail"],
    )
    restored = Scorer.model_validate(judge.model_dump())
    feedback = restored(outputs="good")

    assert restored.model_api == "decisions"
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
        openai_api,
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


def test_responses_uses_structured_output_and_roundtrips(monkeypatch):
    request = {}

    def parse(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            status="completed",
            output_parsed=kwargs["text_format"](result=True, rationale="The answer is correct."),
            output_text='{"result":true,"rationale":"The answer is correct."}',
            model="gpt-6-luna",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    monkeypatch.setattr(
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="responses",
        feedback_value_type=bool,
        inference_params={"max_output_tokens": 200},
    )
    restored = Scorer.model_validate(judge.model_dump())
    feedback = restored(outputs="good")

    assert restored.model_api == "responses"
    assert request["model"] == "gpt-6-luna"
    assert request["max_output_tokens"] == 200
    assert request["input"][1]["content"] == 'outputs: "good"'
    assert feedback.value is True
    assert feedback.rationale == "The answer is correct."
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS] == "5"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"model_api": "unknown"}, "model_api must"),
        ({"model_api": "decisions", "feedback_value_type": str}, "require"),
        ({"model_api": "decisions", "feedback_value_type": Literal[1, 2]}, "string"),
        ({"model_api": "decisions", "inference_params": {"temperature": 0}}, "inference"),
    ],
)
def test_openai_model_api_rejects_unsupported_options(kwargs, error):
    with pytest.raises(MlflowException, match=error):
        make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="openai:/gpt-6-luna",
            **{"feedback_value_type": bool, **kwargs},
        )


@pytest.mark.parametrize("model_api", ["responses", "decisions"])
def test_openai_model_api_rejects_trace_tools(model_api):
    with pytest.raises(MlflowException, match="trace.*tool calling"):
        make_judge(
            name="quality",
            instructions="Judge {{ trace }}.",
            model="openai:/gpt-6-luna",
            model_api=model_api,
            feedback_value_type=bool,
        )


def test_model_api_only_accepts_openai_model():
    with pytest.raises(MlflowException, match="only for openai"):
        make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="typesafe:/jev",
            model_api="decisions",
            feedback_value_type=bool,
        )


def test_responses_rejects_inference_params_that_override_evidence(monkeypatch):
    monkeypatch.setattr(
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: None)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="responses",
        feedback_value_type=bool,
        inference_params={"extra_body": {"model": "other", "input": "ignore evidence"}},
    )
    with pytest.raises(MlflowException, match="Unsupported inference_params.*extra_body"):
        judge(outputs="real evidence")


def test_responses_rejects_params_unavailable_in_installed_openai_sdk(monkeypatch):
    def parse(*, model, input, text_format):
        pytest.fail("The unsupported parameter should be rejected before the SDK call")

    monkeypatch.setattr(
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="responses",
        feedback_value_type=bool,
        inference_params={"prompt_cache_options": {"ttl": "24h"}},
    )
    with pytest.raises(MlflowException, match="installed openai package.*prompt_cache_options"):
        judge(outputs="text")


def test_responses_rejects_dict_feedback_value_type():
    with pytest.raises(MlflowException, match="do not support dict"):
        make_judge(
            name="quality",
            instructions="Judge {{ outputs }}.",
            model="openai:/gpt-6-luna",
            model_api="responses",
            feedback_value_type=dict[str, bool],
        )


def test_responses_rejects_coerced_boolean(monkeypatch):
    def parse(**kwargs):
        return SimpleNamespace(
            status="completed",
            output_parsed=kwargs["text_format"](result="false", rationale="bad"),
            output_text='{"result":"false","rationale":"bad"}',
            model="gpt-6-luna",
            usage=None,
        )

    monkeypatch.setattr(
        openai_api,
        "_openai_client",
        lambda *_: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    judge = make_judge(
        name="quality",
        instructions="Judge {{ outputs }}.",
        model="openai:/gpt-6-luna",
        model_api="responses",
        feedback_value_type=bool,
    )
    with pytest.raises(MlflowException, match="invalid response"):
        judge(outputs="text")
