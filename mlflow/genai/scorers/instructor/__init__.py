"""
Instructor integration for MLflow.

This module provides integration with Instructor for structured output validation,
allowing Pydantic schema validation to be used as MLflow scorers.

Example usage:

.. code-block:: python

    from pydantic import BaseModel
    from mlflow.genai.scorers.instructor import SchemaCompliance


    class UserInfo(BaseModel):
        name: str
        email: str
        age: int


    scorer = SchemaCompliance()
    feedback = scorer(
        outputs={"name": "John", "email": "john@example.com", "age": 30},
        expectations={"schema": UserInfo},
    )
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr, ValidationError

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.instructor.utils import (
    check_instructor_installed,
    get_schema_from_expectations,
    output_to_dict,
    resolve_output_for_validation,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.10.0")
class InstructorScorer(Scorer):
    """
    Base scorer class for Instructor/Pydantic validation metrics.

    All Instructor scorers are deterministic (no LLM required) and validate
    structured outputs against Pydantic schemas.

    Args:
        scorer_name: Name of the scorer. If not provided, uses class-level scorer_name.
    """

    _schema: Any = PrivateAttr(default=None)

    def __init__(self, scorer_name: str | None = None):
        check_instructor_installed()
        if scorer_name is None:
            scorer_name = getattr(self.__class__, "scorer_name", self.__class__.__name__)
        # scorer_name is guaranteed to be str at this point
        assert isinstance(scorer_name, str)
        super().__init__(name=scorer_name)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def _raise_registration_not_supported(self, method_name: str):
        raise MlflowException.invalid_parameter_value(
            f"'{method_name}()' is not supported for third-party scorers like Instructor. "
            f"Third-party scorers cannot be registered, started, updated, or stopped. "
            f"Use them directly in mlflow.genai.evaluate() instead."
        )

    def register(self, **kwargs):
        self._raise_registration_not_supported("register")

    def start(self, **kwargs):
        self._raise_registration_not_supported("start")

    def update(self, **kwargs):
        self._raise_registration_not_supported("update")

    def stop(self, **kwargs):
        self._raise_registration_not_supported("stop")

    def align(self, **kwargs):
        raise MlflowException.invalid_parameter_value(
            "'align()' is not supported for third-party scorers like Instructor. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """
        Evaluate using Pydantic schema validation.

        Args:
            inputs: The input (not used for validation)
            outputs: The output to validate against the schema
            expectations: Must contain 'schema' key with Pydantic model class
            trace: MLflow trace for evaluation
            session: List of MLflow traces (not used for Instructor scorers)

        Returns:
            Feedback object with validation result
        """
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.CODE,
            source_id=f"instructor/{self.name}",
        )

        try:
            return self._evaluate(outputs, expectations, trace, assessment_source)
        except Exception as e:
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        raise NotImplementedError("Subclasses must implement _evaluate")


@experimental(version="3.10.0")
class SchemaCompliance(InstructorScorer):
    """
    Validates that output matches the expected Pydantic schema.

    This scorer checks if the output can be successfully validated against
    the provided Pydantic model, returning YES if valid, NO otherwise.

    Examples:
        .. code-block:: python

            from pydantic import BaseModel
            from mlflow.genai.scorers.instructor import SchemaCompliance


            class UserInfo(BaseModel):
                name: str
                email: str
                age: int


            scorer = SchemaCompliance()
            feedback = scorer(
                outputs={"name": "John", "email": "john@example.com", "age": 30},
                expectations={"schema": UserInfo},
            )
            print(feedback.value)  # CategoricalRating.YES
    """

    scorer_name: ClassVar[str] = "SchemaCompliance"

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        schema = get_schema_from_expectations(expectations)
        outputs = resolve_output_for_validation(outputs, trace)
        output_dict = output_to_dict(outputs)

        try:
            schema.model_validate(output_dict)
            return Feedback(
                name=self.name,
                value=CategoricalRating.YES,
                rationale=None,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )
        except ValidationError as e:
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            return Feedback(
                name=self.name,
                value=CategoricalRating.NO,
                rationale=f"Schema validation failed: {error_details}",
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )


@experimental(version="3.10.0")
class FieldCompleteness(InstructorScorer):
    """
    Checks that all required fields are present and non-null.

    Returns a score between 0.0 and 1.0 representing the percentage of
    required fields that are present and have non-null values.

    Examples:
        .. code-block:: python

            from pydantic import BaseModel
            from mlflow.genai.scorers.instructor import FieldCompleteness


            class UserInfo(BaseModel):
                name: str
                email: str
                age: int


            scorer = FieldCompleteness()
            feedback = scorer(
                outputs={"name": "John", "email": None, "age": 30},
                expectations={"schema": UserInfo},
            )
            print(feedback.value)  # 0.67 (2 out of 3 fields complete)
    """

    scorer_name: ClassVar[str] = "FieldCompleteness"

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        schema = get_schema_from_expectations(expectations)
        outputs = resolve_output_for_validation(outputs, trace)
        output_dict = output_to_dict(outputs)

        # Get required fields from schema
        required_fields = []
        for field_name, field_info in schema.model_fields.items():
            if field_info.is_required():
                required_fields.append(field_name)

        if not required_fields:
            return Feedback(
                name=self.name,
                value=1.0,
                rationale="No required fields in schema",
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )

        # Count complete fields (present and non-null)
        complete_count = 0
        incomplete_fields = []
        for field in required_fields:
            if field in output_dict and output_dict[field] is not None:
                complete_count += 1
            else:
                incomplete_fields.append(field)

        score = complete_count / len(required_fields)

        if incomplete_fields:
            rationale = f"Missing or null fields: {', '.join(incomplete_fields)}"
        else:
            rationale = None

        return Feedback(
            name=self.name,
            value=score,
            rationale=rationale,
            source=assessment_source,
            metadata={FRAMEWORK_METADATA_KEY: "instructor"},
        )


@experimental(version="3.10.0")
class TypeValidation(InstructorScorer):
    """
    Verifies that field types match schema definitions.

    Returns YES if all fields have correct types, NO otherwise.
    Provides detailed error messages for type mismatches.

    Examples:
        .. code-block:: python

            from pydantic import BaseModel
            from mlflow.genai.scorers.instructor import TypeValidation


            class UserInfo(BaseModel):
                name: str
                age: int


            scorer = TypeValidation()
            feedback = scorer(
                outputs={"name": "John", "age": "thirty"},  # age should be int
                expectations={"schema": UserInfo},
            )
            print(feedback.value)  # CategoricalRating.NO
    """

    scorer_name: ClassVar[str] = "TypeValidation"

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        schema = get_schema_from_expectations(expectations)
        outputs = resolve_output_for_validation(outputs, trace)
        output_dict = output_to_dict(outputs)

        try:
            schema.model_validate(output_dict)
            return Feedback(
                name=self.name,
                value=CategoricalRating.YES,
                rationale=None,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )
        except ValidationError as e:
            # Filter for type-related errors
            type_errors = [
                err
                for err in e.errors()
                if err["type"]
                in (
                    "type_error",
                    "int_parsing",
                    "float_parsing",
                    "bool_parsing",
                    "string_type",
                    "int_type",
                    "float_type",
                    "bool_type",
                    "list_type",
                    "dict_type",
                )
                or "type" in err["type"]
            ]

            if type_errors:
                error_details = "; ".join(
                    f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                    for err in type_errors
                )
                return Feedback(
                    name=self.name,
                    value=CategoricalRating.NO,
                    rationale=f"Type validation failed: {error_details}",
                    source=assessment_source,
                    metadata={FRAMEWORK_METADATA_KEY: "instructor"},
                )
            else:
                # No type errors, but other validation errors exist
                return Feedback(
                    name=self.name,
                    value=CategoricalRating.YES,
                    rationale="All field types are valid",
                    source=assessment_source,
                    metadata={FRAMEWORK_METADATA_KEY: "instructor"},
                )


@experimental(version="3.10.0")
class ConstraintValidation(InstructorScorer):
    """
    Checks that Pydantic validators and constraints pass.

    This includes Field constraints (min_length, max_length, ge, le, regex, etc.)
    and custom validators.

    Examples:
        .. code-block:: python

            from pydantic import BaseModel, Field
            from mlflow.genai.scorers.instructor import ConstraintValidation


            class UserInfo(BaseModel):
                name: str = Field(min_length=1, max_length=100)
                age: int = Field(ge=0, le=150)
                email: str = Field(pattern=r"^[\\w.-]+@[\\w.-]+\\.\\w+$")


            scorer = ConstraintValidation()
            feedback = scorer(
                outputs={"name": "", "age": 200, "email": "invalid"},
                expectations={"schema": UserInfo},
            )
            print(feedback.value)  # CategoricalRating.NO
    """

    scorer_name: ClassVar[str] = "ConstraintValidation"

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        schema = get_schema_from_expectations(expectations)
        outputs = resolve_output_for_validation(outputs, trace)
        output_dict = output_to_dict(outputs)

        try:
            schema.model_validate(output_dict)
            return Feedback(
                name=self.name,
                value=CategoricalRating.YES,
                rationale=None,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )
        except ValidationError as e:
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            return Feedback(
                name=self.name,
                value=CategoricalRating.NO,
                rationale=f"Constraint validation failed: {error_details}",
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )


@experimental(version="3.10.0")
class ExtractionAccuracy(InstructorScorer):
    """
    Compares extracted fields against ground truth values.

    Returns a score between 0.0 and 1.0 representing the percentage of
    fields that match the expected values. Requires 'expected_output'
    in expectations.

    Examples:
        .. code-block:: python

            from pydantic import BaseModel
            from mlflow.genai.scorers.instructor import ExtractionAccuracy


            class UserInfo(BaseModel):
                name: str
                age: int


            scorer = ExtractionAccuracy()
            feedback = scorer(
                outputs={"name": "John Doe", "age": 30},
                expectations={
                    "schema": UserInfo,
                    "expected_output": {"name": "John Doe", "age": 25},
                },
            )
            print(feedback.value)  # 0.5 (1 out of 2 fields match)
    """

    scorer_name: ClassVar[str] = "ExtractionAccuracy"

    def _evaluate(
        self,
        outputs: Any,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        assessment_source: AssessmentSource,
    ) -> Feedback:
        schema = get_schema_from_expectations(expectations)
        outputs = resolve_output_for_validation(outputs, trace)
        output_dict = output_to_dict(outputs)

        if not expectations or "expected_output" not in expectations:
            raise MlflowException.invalid_parameter_value(
                "ExtractionAccuracy requires 'expected_output' in expectations. "
                "Provide ground truth values: "
                "expectations={'schema': Model, 'expected_output': {...}}"
            )

        expected = expectations["expected_output"]
        if hasattr(expected, "model_dump"):
            expected = expected.model_dump()
        elif not isinstance(expected, dict):
            raise MlflowException.invalid_parameter_value(
                "'expected_output' must be a dict or Pydantic model instance"
            )

        # Compare fields from schema
        fields = list(schema.model_fields.keys())
        if not fields:
            return Feedback(
                name=self.name,
                value=1.0,
                rationale="No fields to compare",
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "instructor"},
            )

        matching_count = 0
        mismatched_fields = []
        for field in fields:
            actual_value = output_dict.get(field)
            expected_value = expected.get(field)
            if actual_value == expected_value:
                matching_count += 1
            else:
                mismatched_fields.append(
                    f"{field}: expected={expected_value!r}, got={actual_value!r}"
                )

        score = matching_count / len(fields)

        if mismatched_fields:
            rationale = f"Field mismatches: {'; '.join(mismatched_fields)}"
        else:
            rationale = None

        return Feedback(
            name=self.name,
            value=score,
            rationale=rationale,
            source=assessment_source,
            metadata={FRAMEWORK_METADATA_KEY: "instructor"},
        )


@experimental(version="3.10.0")
def get_scorer(scorer_name: str) -> InstructorScorer:
    """
    Get an Instructor scorer by name.

    Args:
        scorer_name: Name of the scorer (e.g., "SchemaCompliance", "FieldCompleteness")

    Returns:
        InstructorScorer instance

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.instructor import get_scorer

            scorer = get_scorer("SchemaCompliance")
            feedback = scorer(
                outputs={"name": "John"},
                expectations={"schema": UserInfo},
            )
    """
    _SCORER_REGISTRY = {
        "SchemaCompliance": SchemaCompliance,
        "FieldCompleteness": FieldCompleteness,
        "TypeValidation": TypeValidation,
        "ConstraintValidation": ConstraintValidation,
        "ExtractionAccuracy": ExtractionAccuracy,
    }

    if scorer_name not in _SCORER_REGISTRY:
        available = ", ".join(sorted(_SCORER_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Instructor scorer: '{scorer_name}'. Available: {available}"
        )

    return _SCORER_REGISTRY[scorer_name]()


__all__ = [
    # Core classes
    "InstructorScorer",
    "get_scorer",
    # Validation scorers
    "SchemaCompliance",
    "FieldCompleteness",
    "TypeValidation",
    "ConstraintValidation",
    "ExtractionAccuracy",
]
