from unittest.mock import patch

import guardrails
import pytest
from guardrails import Validator, register_validator
from guardrails.classes.validation.validation_result import FailResult, PassResult

from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType


@register_validator(name="test/mock_validator", data_type="string")
class MockValidator(Validator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self, value, metadata=None):
        if "toxic" in str(value).lower() or "bad" in str(value).lower():
            return FailResult(error_message="Content flagged as inappropriate")
        return PassResult()


@pytest.fixture
def mock_validator_class():
    return MockValidator


@pytest.mark.parametrize(
    ("scorer_class", "validator_name"),
    [
        ("ToxicLanguage", "ToxicLanguage"),
        ("NSFWText", "NSFWText"),
        ("DetectJailbreak", "DetectJailbreak"),
        ("DetectPII", "DetectPII"),
        ("SecretsPresent", "SecretsPresent"),
        ("GibberishText", "GibberishText"),
    ],
)
def test_guardrails_scorer_pass(mock_validator_class, scorer_class, validator_name):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers import guardrails as guardrails_scorers

        scorer_cls = getattr(guardrails_scorers, scorer_class)
        scorer = scorer_cls()

        assert isinstance(scorer._guard, guardrails.Guard)

        result = scorer(outputs="This is clean text.")

    assert result.name == validator_name
    assert result.value == "pass"
    assert result.rationale is None
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id=f"guardrails/{validator_name}",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails"}


@pytest.mark.parametrize(
    ("scorer_class", "validator_name"),
    [
        ("ToxicLanguage", "ToxicLanguage"),
        ("DetectPII", "DetectPII"),
    ],
)
def test_guardrails_scorer_fail(mock_validator_class, scorer_class, validator_name):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers import guardrails as guardrails_scorers

        scorer_cls = getattr(guardrails_scorers, scorer_class)
        scorer = scorer_cls()
        result = scorer(outputs="This is toxic bad content.")

    assert result.name == validator_name
    assert result.value == "fail"
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id=f"guardrails/{validator_name}",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails"}
    assert "Content flagged as inappropriate" in result.rationale


def test_guardrails_get_scorer(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers.guardrails import get_scorer

        scorer = get_scorer("ToxicLanguage", threshold=0.8)
        result = scorer(outputs="Clean text")

    assert result.name == "ToxicLanguage"
    assert result.value == "pass"
    assert result.rationale is None
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails"}


def test_guardrails_scorer_with_custom_kwargs(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers.guardrails import ToxicLanguage

        scorer = ToxicLanguage(threshold=0.9, validation_method="full")
        result = scorer(outputs="Test text")

    assert result.value == "pass"
    assert result.metadata == {"mlflow.scorer.framework": "guardrails"}


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected_value"),
    [
        ("Input text", "Output text", "pass"),
        ("Input only", None, "pass"),
        ("toxic input", "clean output", "pass"),  # outputs takes priority
        ("clean input", "toxic output", "fail"),  # outputs takes priority
    ],
)
def test_guardrails_scorer_input_priority(mock_validator_class, inputs, outputs, expected_value):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers.guardrails import ToxicLanguage

        scorer = ToxicLanguage()
        result = scorer(inputs=inputs, outputs=outputs)

    assert result.value == expected_value


def test_guardrails_scorer_error_handling(mock_validator_class):
    @register_validator(name="test/error_validator", data_type="string")
    class ErrorValidator(Validator):
        def validate(self, value, metadata=None):
            raise RuntimeError("Validation failed")

    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=ErrorValidator,
    ):
        from mlflow.genai.scorers.guardrails import ToxicLanguage

        scorer = ToxicLanguage()
        result = scorer(outputs="Some text")

    assert result.error is not None
    assert "Validation failed" in str(result.error)
    assert result.source.source_id == "guardrails/ToxicLanguage"
    assert result.metadata == {"mlflow.scorer.framework": "guardrails"}


def test_guardrails_scorer_source_id(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers.guardrails import ToxicLanguage

        scorer = ToxicLanguage()
        feedback = scorer(outputs="Test")

    assert feedback.source.source_id == "guardrails/ToxicLanguage"
    assert feedback.source.source_type == "CODE"


def test_guardrails_scorer_guard_is_real_instance(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        from mlflow.genai.scorers.guardrails import ToxicLanguage

        scorer = ToxicLanguage()

    assert isinstance(scorer._guard, guardrails.Guard)
