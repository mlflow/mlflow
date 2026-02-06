import instructor  # noqa: F401 - Import at top level, tests fail if not installed
import pytest
from pydantic import BaseModel, Field

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.instructor import (
    ConstraintValidation,
    ExtractionAccuracy,
    FieldCompleteness,
    SchemaCompliance,
    TypeValidation,
    get_scorer,
)


class UserInfo(BaseModel):
    name: str
    email: str
    age: int


class UserInfoWithConstraints(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r"^[\w.-]+@[\w.-]+\.\w+$")


class UserInfoOptional(BaseModel):
    name: str
    email: str | None = None
    age: int | None = None


@pytest.mark.parametrize(
    ("scorer_class", "scorer_name"),
    [
        (SchemaCompliance, "SchemaCompliance"),
        (FieldCompleteness, "FieldCompleteness"),
        (TypeValidation, "TypeValidation"),
        (ConstraintValidation, "ConstraintValidation"),
        (ExtractionAccuracy, "ExtractionAccuracy"),
    ],
)
def test_instructor_scorer_instantiation(scorer_class, scorer_name):
    scorer = scorer_class()
    assert scorer.name == scorer_name


@pytest.mark.parametrize(
    ("outputs", "expected_value", "rationale_contains"),
    [
        # Pass case
        (
            {"name": "John Doe", "email": "john@example.com", "age": 30},
            CategoricalRating.YES,
            None,
        ),
        # Fail case - missing field
        (
            {"name": "John Doe", "email": "john@example.com"},
            CategoricalRating.NO,
            "age",
        ),
        # Fail case - type error
        (
            {"name": "John", "email": "john@example.com", "age": "thirty"},
            CategoricalRating.NO,
            "age",
        ),
    ],
)
def test_schema_compliance(outputs, expected_value, rationale_contains):
    scorer = SchemaCompliance()
    result = scorer(outputs=outputs, expectations={"schema": UserInfo})

    assert isinstance(result, Feedback)
    assert result.name == "SchemaCompliance"
    assert result.value == expected_value
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="instructor/SchemaCompliance",
    )
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}
    if rationale_contains:
        assert rationale_contains in result.rationale
    else:
        assert result.rationale is None


@pytest.mark.parametrize(
    ("outputs", "schema", "expected_value", "rationale_fields"),
    [
        # All complete
        (
            {"name": "John", "email": "john@example.com", "age": 30},
            UserInfo,
            1.0,
            None,
        ),
        # Partial - null field
        (
            {"name": "John", "email": None, "age": 30},
            UserInfo,
            pytest.approx(2.0 / 3.0),
            ["email"],
        ),
        # Missing fields
        (
            {"name": "John"},
            UserInfo,
            pytest.approx(1.0 / 3.0),
            ["email", "age"],
        ),
        # Optional fields - only required field present
        (
            {"name": "John"},
            UserInfoOptional,
            1.0,
            None,
        ),
    ],
)
def test_field_completeness(outputs, schema, expected_value, rationale_fields):
    scorer = FieldCompleteness()
    result = scorer(outputs=outputs, expectations={"schema": schema})

    assert isinstance(result, Feedback)
    assert result.name == "FieldCompleteness"
    assert result.value == expected_value
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="instructor/FieldCompleteness",
    )
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}
    if rationale_fields:
        for field in rationale_fields:
            assert field in result.rationale
    else:
        assert result.rationale is None


@pytest.mark.parametrize(
    ("outputs", "expected_value", "rationale_contains"),
    [
        # Pass case
        (
            {"name": "John", "email": "john@example.com", "age": 30},
            CategoricalRating.YES,
            None,
        ),
        # Fail case - wrong type
        (
            {"name": "John", "email": "john@example.com", "age": "thirty"},
            CategoricalRating.NO,
            "age",
        ),
    ],
)
def test_type_validation(outputs, expected_value, rationale_contains):
    scorer = TypeValidation()
    result = scorer(outputs=outputs, expectations={"schema": UserInfo})

    assert isinstance(result, Feedback)
    assert result.name == "TypeValidation"
    assert result.value == expected_value
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="instructor/TypeValidation",
    )
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}
    if rationale_contains:
        assert rationale_contains in result.rationale


@pytest.mark.parametrize(
    ("outputs", "expected_value"),
    [
        # Pass case
        ({"name": "John", "age": 30, "email": "john@example.com"}, CategoricalRating.YES),
        # Fail case - constraint violations
        ({"name": "", "age": 200, "email": "invalid"}, CategoricalRating.NO),
    ],
)
def test_constraint_validation(outputs, expected_value):
    scorer = ConstraintValidation()
    result = scorer(outputs=outputs, expectations={"schema": UserInfoWithConstraints})

    assert isinstance(result, Feedback)
    assert result.name == "ConstraintValidation"
    assert result.value == expected_value
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="instructor/ConstraintValidation",
    )
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}


@pytest.mark.parametrize(
    ("outputs", "expected_output", "expected_value", "rationale_contains"),
    [
        # Perfect match
        (
            {"name": "John Doe", "email": "john@example.com", "age": 30},
            {"name": "John Doe", "email": "john@example.com", "age": 30},
            1.0,
            None,
        ),
        # Partial match - age differs
        (
            {"name": "John Doe", "email": "john@example.com", "age": 30},
            {"name": "John Doe", "email": "john@example.com", "age": 25},
            pytest.approx(2.0 / 3.0),
            "age",
        ),
    ],
)
def test_extraction_accuracy(outputs, expected_output, expected_value, rationale_contains):
    scorer = ExtractionAccuracy()
    result = scorer(
        outputs=outputs,
        expectations={"schema": UserInfo, "expected_output": expected_output},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExtractionAccuracy"
    assert result.value == expected_value
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="instructor/ExtractionAccuracy",
    )
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}
    if rationale_contains:
        assert rationale_contains in result.rationale
    else:
        assert result.rationale is None


def test_extraction_accuracy_missing_expected_output():
    scorer = ExtractionAccuracy()
    result = scorer(
        outputs={"name": "John", "email": "john@example.com", "age": 30},
        expectations={"schema": UserInfo},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExtractionAccuracy"
    assert result.error is not None
    assert "expected_output" in str(result.error)
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}


def test_get_scorer():
    scorer = get_scorer("SchemaCompliance")
    result = scorer(
        outputs={"name": "John", "email": "john@example.com", "age": 30},
        expectations={"schema": UserInfo},
    )

    assert isinstance(result, Feedback)
    assert result.name == "SchemaCompliance"
    assert result.value == CategoricalRating.YES


def test_get_scorer_unknown():
    with pytest.raises(MlflowException, match="Unknown Instructor scorer"):
        get_scorer("UnknownScorer")


@pytest.mark.parametrize(
    ("expectations", "error_contains"),
    [
        ({}, "schema"),
        ({"schema": "not_a_class"}, "Pydantic model class"),
    ],
)
def test_scorer_invalid_expectations(expectations, error_contains):
    scorer = SchemaCompliance()
    result = scorer(outputs={"name": "John"}, expectations=expectations)

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert error_contains in str(result.error)
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}


@pytest.mark.parametrize(
    "outputs",
    [
        {"name": "John", "email": "john@example.com", "age": 30},  # dict
        UserInfo(name="John", email="john@example.com", age=30),  # Pydantic model
        '{"name": "John", "email": "john@example.com", "age": 30}',  # JSON string
    ],
)
def test_scorer_output_formats(outputs):
    scorer = SchemaCompliance()
    result = scorer(outputs=outputs, expectations={"schema": UserInfo})

    assert isinstance(result, Feedback)
    assert result.value == CategoricalRating.YES


def _create_test_trace(inputs=None, outputs=None):
    with mlflow.start_span() as span:
        if inputs is not None:
            span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


@pytest.mark.parametrize(
    ("trace_outputs", "expected_value"),
    [
        ({"name": "John", "email": "john@example.com", "age": 30}, CategoricalRating.YES),
        ({"name": "John"}, CategoricalRating.NO),  # missing fields
    ],
)
def test_scorer_with_trace(trace_outputs, expected_value):
    trace = _create_test_trace(
        inputs={"query": "Get user info"},
        outputs=trace_outputs,
    )

    scorer = SchemaCompliance()
    result = scorer(trace=trace, expectations={"schema": UserInfo})

    assert isinstance(result, Feedback)
    assert result.name == "SchemaCompliance"
    assert result.value == expected_value


@pytest.mark.parametrize(
    "scorer_class",
    [
        SchemaCompliance,
        FieldCompleteness,
        TypeValidation,
        ConstraintValidation,
        ExtractionAccuracy,
    ],
)
def test_scorer_error_includes_framework_metadata(scorer_class):
    scorer = scorer_class()
    result = scorer(outputs={"name": "John"}, expectations={})

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert result.metadata == {"mlflow.scorer.framework": "instructor"}
