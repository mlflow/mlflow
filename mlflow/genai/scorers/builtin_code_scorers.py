import json
import math
import re
from dataclasses import asdict
from typing import Any, Literal

import pydantic

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import (
    _SERIALIZATION_VERSION,
    Scorer,
    ScorerKind,
    SerializedScorer,
)

# Common PII regex patterns
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


def _passes_luhn(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class BuiltInCodeScorer(Scorer):
    """Base class for built-in deterministic code scorers.

    These scorers run without LLM calls and produce instant, reproducible results.
    They complement the existing LLM-based judges by providing structural and
    safety checks on model outputs.
    """

    name: str
    required_columns: set[str] = set()

    def validate_columns(self, columns: set[str]) -> None:
        missing = self.required_columns - columns
        if missing:
            raise MlflowException.invalid_parameter_value(
                f"Scorer '{self.name}' requires columns {missing} not present in dataset."
            )

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def _make_feedback(self, *, value: bool, rationale: str) -> Feedback:
        return Feedback(
            name=self.name,
            value=value,
            rationale=rationale,
            source=AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id=f"mlflow.scorers.{self.__class__.__name__}",
            ),
        )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        pydantic_model_data = pydantic.BaseModel.model_dump(self, mode="json", **kwargs)

        serialized = SerializedScorer(
            name=self.name,
            description=self.description,
            aggregations=self.aggregations,
            is_session_level_scorer=self.is_session_level_scorer,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            builtin_scorer_class=self.__class__.__name__,
            builtin_scorer_pydantic_data=pydantic_model_data,
        )

        return asdict(serialized)

    @classmethod
    def model_validate(cls, obj: SerializedScorer | dict[str, Any]) -> "BuiltInCodeScorer":
        from mlflow.genai.scorers import builtin_code_scorers

        if isinstance(obj, SerializedScorer):
            serialized = obj
        else:
            if not isinstance(obj, dict) or "builtin_scorer_class" not in obj:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid builtin code scorer data: expected a dictionary with "
                    f"'builtin_scorer_class' field, got {type(obj).__name__}."
                )
            try:
                serialized = SerializedScorer(**obj)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to parse serialized scorer data: {e}"
                )

        try:
            scorer_class = getattr(builtin_code_scorers, serialized.builtin_scorer_class)
        except AttributeError:
            raise MlflowException.invalid_parameter_value(
                f"Unknown builtin code scorer class: {serialized.builtin_scorer_class}"
            )

        constructor_args = serialized.builtin_scorer_pydantic_data or {}
        return scorer_class(**constructor_args)


class ExactMatch(BuiltInCodeScorer):
    """Check whether the output exactly matches the expected response.

    Args:
        name: The name of the scorer. Defaults to ``"exact_match"``.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import ExactMatch

        scorer = ExactMatch()
        feedback = scorer(outputs="hello", expectations={"expected_response": "hello"})
        print(feedback.value)  # True
    """

    name: str = "exact_match"
    required_columns: set[str] = {"outputs", "expectations"}

    def __call__(
        self,
        *,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
    ) -> Feedback:
        output_str = _to_str(outputs)
        expected = ""
        if isinstance(expectations, dict):
            expected = _to_str(expectations.get("expected_response", ""))
        else:
            expected = _to_str(expectations)

        match = output_str == expected
        rationale = (
            "Output matches expected response."
            if match
            else f"Output does not match expected response. "
            f"Expected: {expected!r}, Got: {output_str!r}"
        )
        return self._make_feedback(value=match, rationale=rationale)


class JsonValidity(BuiltInCodeScorer):
    """Check whether the output is valid JSON, optionally with required keys.

    Args:
        name: The name of the scorer. Defaults to ``"json_validity"``.
        required_keys: Optional list of top-level keys that must be present
            in the parsed JSON object.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import JsonValidity

        scorer = JsonValidity(required_keys=["answer", "sources"])
        feedback = scorer(outputs='{"answer": "yes", "sources": []}')
        print(feedback.value)  # True
    """

    name: str = "json_validity"
    required_keys: list[str] | None = None

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return self._make_feedback(
                value=False,
                rationale=f"Output is not valid JSON: {text[:200]!r}",
            )

        if self.required_keys:
            if not isinstance(parsed, dict):
                return self._make_feedback(
                    value=False,
                    rationale=f"JSON output is not an object (got {type(parsed).__name__}), "
                    f"cannot check required keys.",
                )
            missing = [k for k in self.required_keys if k not in parsed]
            if missing:
                return self._make_feedback(
                    value=False,
                    rationale=f"JSON output is missing required keys: {missing}",
                )

        return self._make_feedback(
            value=True,
            rationale="Output is valid JSON"
            + (f" with all required keys: {self.required_keys}" if self.required_keys else "")
            + ".",
        )


class RegexMatch(BuiltInCodeScorer):
    """Check whether the output matches a regex pattern.

    Note:
        Ensure regex patterns are efficient. Patterns with nested quantifiers
        (e.g., ``(a+)+b``) may cause slow execution on large outputs.

    Args:
        name: The name of the scorer. Defaults to ``"regex_match"``.
        pattern: The regex pattern to search for in the output.
        full_match: If ``True``, require the entire output to match the pattern (using
            ``re.fullmatch``). If ``False`` (default), search for the pattern anywhere
            in the output (using ``re.search``).

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import RegexMatch

        scorer = RegexMatch(pattern=r"\\d{3}-\\d{4}")
        feedback = scorer(outputs="Call 555-1234 for info")
        print(feedback.value)  # True
    """

    name: str = "regex_match"
    pattern: str
    full_match: bool = False

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)

        try:
            if self.full_match:
                matched = bool(re.fullmatch(self.pattern, text))
            else:
                matched = bool(re.search(self.pattern, text))
        except re.error as e:
            return self._make_feedback(
                value=False,
                rationale=f"Invalid regex pattern {self.pattern!r}: {e}",
            )

        rationale = (
            f"Output {'fully matches' if self.full_match else 'matches'} pattern {self.pattern!r}."
            if matched
            else f"Output does not match pattern {self.pattern!r}."
        )
        return self._make_feedback(value=matched, rationale=rationale)


class ContainsKeywords(BuiltInCodeScorer):
    """Check whether the output contains all required keywords or phrases.

    Args:
        name: The name of the scorer. Defaults to ``"contains_keywords"``.
        keywords: List of keywords or phrases that must appear in the output.
        case_sensitive: Whether the keyword check is case-sensitive. Defaults to ``False``.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["disclaimer", "not financial advice"])
        feedback = scorer(outputs="Disclaimer: This is not financial advice.")
        print(feedback.value)  # True
    """

    name: str = "contains_keywords"
    keywords: list[str]
    case_sensitive: bool = False

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)
        check_text = text if self.case_sensitive else text.lower()

        missing = []
        for keyword in self.keywords:
            check_keyword = keyword if self.case_sensitive else keyword.lower()
            if check_keyword not in check_text:
                missing.append(keyword)

        if missing:
            return self._make_feedback(
                value=False,
                rationale=f"Output is missing keywords: {missing}",
            )
        return self._make_feedback(
            value=True,
            rationale=f"Output contains all required keywords: {self.keywords}",
        )


class LengthBound(BuiltInCodeScorer):
    """Check whether the output length is within specified bounds.

    Args:
        name: The name of the scorer. Defaults to ``"length_bound"``.
        min_length: Minimum allowed length (inclusive). Defaults to ``None`` (no minimum).
        max_length: Maximum allowed length (inclusive). Defaults to ``None`` (no maximum).
        unit: Unit of measurement: ``"chars"`` or ``"words"``. Defaults to ``"chars"``.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import LengthBound

        scorer = LengthBound(min_length=10, max_length=500, unit="chars")
        feedback = scorer(outputs="A short response.")
        print(feedback.value)  # True
    """

    name: str = "length_bound"
    min_length: int | None = None
    max_length: int | None = None
    unit: Literal["chars", "words"] = "chars"

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)

        length = len(text.split()) if self.unit == "words" else len(text)

        violations = []
        if self.min_length is not None and length < self.min_length:
            violations.append(f"below minimum of {self.min_length}")
        if self.max_length is not None and length > self.max_length:
            violations.append(f"above maximum of {self.max_length}")

        if violations:
            return self._make_feedback(
                value=False,
                rationale=f"Output length {length} {self.unit} is "
                + " and ".join(violations)
                + ".",
            )
        return self._make_feedback(
            value=True,
            rationale=f"Output length {length} {self.unit} is within bounds"
            + (f" (min={self.min_length})" if self.min_length is not None else "")
            + (f" (max={self.max_length})" if self.max_length is not None else "")
            + ".",
        )


class IsNotEmpty(BuiltInCodeScorer):
    """Check whether the output is non-empty and non-whitespace.

    Args:
        name: The name of the scorer. Defaults to ``"is_not_empty"``.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import IsNotEmpty

        scorer = IsNotEmpty()
        feedback = scorer(outputs="Hello, world!")
        print(feedback.value)  # True
    """

    name: str = "is_not_empty"

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)
        is_non_empty = bool(text.strip())

        return self._make_feedback(
            value=is_non_empty,
            rationale="Output is non-empty."
            if is_non_empty
            else "Output is empty or contains only whitespace.",
        )


class LatencyThreshold(BuiltInCodeScorer):
    """Check whether the trace execution latency is under a threshold.

    This scorer requires a trace object to extract the execution duration.

    Args:
        name: The name of the scorer. Defaults to ``"latency_threshold"``.
        max_latency_seconds: Maximum allowed latency in seconds.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import LatencyThreshold

        scorer = LatencyThreshold(max_latency_seconds=3.0)
        feedback = scorer(trace=my_trace)
        print(feedback.value)  # True if latency < 3s
    """

    name: str = "latency_threshold"
    max_latency_seconds: float
    required_columns: set[str] = {"trace"}

    def __call__(self, *, trace: Trace | None = None) -> Feedback:
        if trace is None or trace.info.execution_duration is None:
            return self._make_feedback(
                value=False,
                rationale="No trace or execution duration available to measure latency.",
            )

        latency_seconds = trace.info.execution_duration / 1000.0
        within_threshold = latency_seconds <= self.max_latency_seconds

        return self._make_feedback(
            value=within_threshold,
            rationale=f"Execution latency {latency_seconds:.3f}s is "
            + (
                f"within threshold of {self.max_latency_seconds}s."
                if within_threshold
                else f"above threshold of {self.max_latency_seconds}s."
            ),
        )


class NumericBound(BuiltInCodeScorer):
    """Check whether a numeric output is within a specified range.

    Args:
        name: The name of the scorer. Defaults to ``"numeric_bound"``.
        min_value: Minimum allowed value (inclusive). Defaults to ``None`` (no minimum).
        max_value: Maximum allowed value (inclusive). Defaults to ``None`` (no maximum).

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0)
        feedback = scorer(outputs=0.85)
        print(feedback.value)  # True
    """

    name: str = "numeric_bound"
    min_value: float | None = None
    max_value: float | None = None

    def __call__(self, *, outputs: Any = None) -> Feedback:
        try:
            numeric_value = float(outputs)
        except (TypeError, ValueError):
            return self._make_feedback(
                value=False,
                rationale=f"Output {outputs!r} cannot be converted to a number.",
            )

        if not math.isfinite(numeric_value):
            return self._make_feedback(
                value=False,
                rationale=f"Output {outputs!r} is not a finite number.",
            )

        violations = []
        if self.min_value is not None and numeric_value < self.min_value:
            violations.append(f"below minimum of {self.min_value}")
        if self.max_value is not None and numeric_value > self.max_value:
            violations.append(f"above maximum of {self.max_value}")

        if violations:
            return self._make_feedback(
                value=False,
                rationale=f"Numeric value {numeric_value} is " + " and ".join(violations) + ".",
            )
        return self._make_feedback(
            value=True,
            rationale=f"Numeric value {numeric_value} is within bounds"
            + (f" (min={self.min_value})" if self.min_value is not None else "")
            + (f" (max={self.max_value})" if self.max_value is not None else "")
            + ".",
        )


class PII(BuiltInCodeScorer):
    """Check whether the output contains personally identifiable information (PII).

    Detects common PII patterns using regex: email addresses, phone numbers,
    Social Security numbers, and credit card numbers. Returns ``True`` (pass) when
    **no** PII is detected, and ``False`` (fail) when PII is found.

    Args:
        name: The name of the scorer. Defaults to ``"pii_detection"``.
        pii_types: Optional list of PII types to check. Valid values are
            ``"email"``, ``"phone"``, ``"ssn"``, ``"credit_card"``.
            Defaults to all types.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PII

        scorer = PII()
        feedback = scorer(outputs="Contact me at user@example.com")
        print(feedback.value)  # False (PII detected)
    """

    name: str = "pii_detection"
    pii_types: list[Literal["email", "phone", "ssn", "credit_card"]] | None = None

    def __call__(self, *, outputs: Any = None) -> Feedback:
        text = _to_str(outputs)

        patterns = _PII_PATTERNS
        if self.pii_types:
            patterns = {k: v for k, v in _PII_PATTERNS.items() if k in self.pii_types}

        detected = {}
        for pii_type, pattern in patterns.items():
            matches = pattern.findall(text)
            if pii_type == "credit_card":
                matches = [m for m in matches if _passes_luhn(m)]
            if matches:
                detected[pii_type] = len(matches)

        if detected:
            details = ", ".join(f"{count} {pii_type}" for pii_type, count in detected.items())
            return self._make_feedback(
                value=False,
                rationale=f"PII detected in output: {details}.",
            )
        return self._make_feedback(
            value=True,
            rationale="No PII detected in output.",
        )
