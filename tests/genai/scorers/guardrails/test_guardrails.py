from unittest.mock import patch

import guardrails
import pytest
from guardrails import Validator, register_validator
from guardrails.classes.validation.validation_result import FailResult, PassResult

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.guardrails import (
    DetectJailbreak,
    DetectPII,
    GibberishText,
    NSFWText,
    SecretsPresent,
    ToxicLanguage,
    get_scorer,
)


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
        (ToxicLanguage, "ToxicLanguage"),
        (NSFWText, "NSFWText"),
        (DetectJailbreak, "DetectJailbreak"),
        (DetectPII, "DetectPII"),
        (SecretsPresent, "SecretsPresent"),
        (GibberishText, "GibberishText"),
    ],
)
def test_guardrails_scorer_pass(mock_validator_class, scorer_class, validator_name):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = scorer_class()

        assert isinstance(scorer._guard, guardrails.Guard)

        result = scorer(outputs="This is clean text.")

    assert isinstance(result, Feedback)
    assert result.name == validator_name
    assert result.value == CategoricalRating.YES
    assert result.rationale is None
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id=f"guardrails/{validator_name}",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}


@pytest.mark.parametrize(
    ("scorer_class", "validator_name"),
    [
        (ToxicLanguage, "ToxicLanguage"),
        (DetectPII, "DetectPII"),
    ],
)
def test_guardrails_scorer_fail(mock_validator_class, scorer_class, validator_name):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = scorer_class()
        result = scorer(outputs="This is toxic bad content.")

    assert isinstance(result, Feedback)
    assert result.name == validator_name
    assert result.value == CategoricalRating.NO
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id=f"guardrails/{validator_name}",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}
    assert "Content flagged as inappropriate" in result.rationale


def test_guardrails_get_scorer(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = get_scorer("ToxicLanguage", threshold=0.8)
        result = scorer(outputs="Clean text")

    assert isinstance(result, Feedback)
    assert result.name == "ToxicLanguage"
    assert result.value == CategoricalRating.YES
    assert result.rationale is None
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}


def test_guardrails_scorer_with_custom_kwargs(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage(threshold=0.9, validation_method="full")
        result = scorer(outputs="Test text")

    assert isinstance(result, Feedback)
    assert result.name == "ToxicLanguage"
    assert result.value == CategoricalRating.YES
    assert result.rationale is None
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected_value"),
    [
        ("Input text", "Output text", CategoricalRating.YES),
        ("Input only", None, CategoricalRating.YES),
        ("toxic input", "clean output", CategoricalRating.YES),  # outputs takes priority
        ("clean input", "toxic output", CategoricalRating.NO),  # outputs takes priority
    ],
)
def test_guardrails_scorer_input_priority(mock_validator_class, inputs, outputs, expected_value):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage()
        result = scorer(inputs=inputs, outputs=outputs)

    assert result.value == expected_value


def test_guardrails_scorer_error_handling():
    @register_validator(name="test/error_validator", data_type="string")
    class ErrorValidator(Validator):
        def validate(self, value, metadata=None):
            raise RuntimeError("Validation failed")

    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=ErrorValidator,
    ):
        scorer = ToxicLanguage()
        result = scorer(outputs="Some text")

    assert isinstance(result, Feedback)
    assert result.name == "ToxicLanguage"
    assert result.error is not None
    assert "Validation failed" in str(result.error)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}


def test_guardrails_scorer_source_id(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage()
        result = scorer(outputs="Test")

    assert isinstance(result, Feedback)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )


def test_guardrails_scorer_guard_is_real_instance(mock_validator_class):
    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage()

    assert isinstance(scorer._guard, guardrails.Guard)


def _create_test_trace(inputs=None, outputs=None):
    """Create a test trace using mlflow.start_span()."""
    with mlflow.start_span() as span:
        if inputs is not None:
            span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


def test_guardrails_scorer_with_trace(mock_validator_class):
    trace = _create_test_trace(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a clean ML platform."},
    )

    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage()
        result = scorer(trace=trace)

    assert isinstance(result, Feedback)
    assert result.name == "ToxicLanguage"
    assert result.value == CategoricalRating.YES
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="guardrails/ToxicLanguage",
    )
    assert result.metadata == {"mlflow.scorer.framework": "guardrails-ai"}


def test_guardrails_scorer_with_trace_failure(mock_validator_class):
    trace = _create_test_trace(
        inputs={"question": "What is toxic?"},
        outputs={"answer": "This is toxic content."},
    )

    with patch(
        "mlflow.genai.scorers.guardrails.get_validator_class",
        return_value=mock_validator_class,
    ):
        scorer = ToxicLanguage()
        result = scorer(trace=trace)

    assert isinstance(result, Feedback)
    assert result.name == "ToxicLanguage"
    assert result.value == CategoricalRating.NO
    assert "Content flagged as inappropriate" in result.rationale
