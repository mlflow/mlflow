from unittest.mock import MagicMock

import pytest

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import (
    PII,
    ContainsKeywords,
    ExactMatch,
    IsNotEmpty,
    JsonValidity,
    LatencyThreshold,
    LengthBound,
    NumericBound,
    RegexMatch,
)
from mlflow.genai.scorers.base import ScorerKind

# ── ExactMatch ──────────────────────────────────────────────────────────────


class TestExactMatch:
    def test_match(self):
        scorer = ExactMatch()
        fb = scorer(outputs="hello", expectations={"expected_response": "hello"})
        assert fb.value is True
        assert "matches" in fb.rationale

    def test_no_match(self):
        scorer = ExactMatch()
        fb = scorer(outputs="hi", expectations={"expected_response": "hello"})
        assert fb.value is False
        assert "does not match" in fb.rationale

    def test_none_outputs(self):
        scorer = ExactMatch()
        fb = scorer(outputs=None, expectations={"expected_response": ""})
        assert fb.value is True

    def test_none_expectations(self):
        scorer = ExactMatch()
        fb = scorer(outputs="hello", expectations=None)
        assert fb.value is False

    def test_non_dict_expectations(self):
        scorer = ExactMatch()
        fb = scorer(outputs="hello", expectations="hello")
        assert fb.value is True

    def test_feedback_source(self):
        scorer = ExactMatch()
        fb = scorer(outputs="a", expectations={"expected_response": "a"})
        assert fb.source.source_type == AssessmentSourceType.CODE
        assert "ExactMatch" in fb.source.source_id

    def test_kind(self):
        assert ExactMatch().kind == ScorerKind.BUILTIN

    def test_name_default(self):
        assert ExactMatch().name == "exact_match"

    def test_custom_name(self):
        assert ExactMatch(name="my_match").name == "my_match"


# ── JsonValidity ────────────────────────────────────────────────────────────


class TestJsonValidity:
    def test_valid_json_object(self):
        fb = JsonValidity()(outputs='{"key": "value"}')
        assert fb.value is True

    def test_valid_json_array(self):
        fb = JsonValidity()(outputs="[1, 2, 3]")
        assert fb.value is True

    def test_invalid_json(self):
        fb = JsonValidity()(outputs="not json")
        assert fb.value is False
        assert "not valid JSON" in fb.rationale

    def test_empty_string(self):
        fb = JsonValidity()(outputs="")
        assert fb.value is False

    def test_required_keys_present(self):
        fb = JsonValidity(required_keys=["a", "b"])(outputs='{"a": 1, "b": 2, "c": 3}')
        assert fb.value is True
        assert "required keys" in fb.rationale

    def test_required_keys_missing(self):
        fb = JsonValidity(required_keys=["a", "b"])(outputs='{"a": 1}')
        assert fb.value is False
        assert "missing required keys" in fb.rationale
        assert "b" in fb.rationale

    def test_required_keys_not_object(self):
        fb = JsonValidity(required_keys=["a"])(outputs="[1, 2]")
        assert fb.value is False
        assert "not an object" in fb.rationale

    def test_none_outputs(self):
        fb = JsonValidity()(outputs=None)
        assert fb.value is False


# ── RegexMatch ──────────────────────────────────────────────────────────────


class TestRegexMatch:
    def test_search_match(self):
        fb = RegexMatch(pattern=r"\d{3}-\d{4}")(outputs="Call 555-1234")
        assert fb.value is True

    def test_search_no_match(self):
        fb = RegexMatch(pattern=r"\d{3}-\d{4}")(outputs="no numbers here")
        assert fb.value is False

    def test_full_match(self):
        fb = RegexMatch(pattern=r"\d+", full_match=True)(outputs="12345")
        assert fb.value is True

    def test_full_match_fails(self):
        fb = RegexMatch(pattern=r"\d+", full_match=True)(outputs="abc12345")
        assert fb.value is False

    def test_invalid_regex(self):
        fb = RegexMatch(pattern=r"[")(outputs="test")
        assert fb.value is False
        assert "Invalid regex" in fb.rationale

    def test_none_outputs(self):
        fb = RegexMatch(pattern=r".")(outputs=None)
        assert fb.value is False


# ── ContainsKeywords ────────────────────────────────────────────────────────


class TestContainsKeywords:
    def test_all_present(self):
        fb = ContainsKeywords(keywords=["hello", "world"])(outputs="hello world")
        assert fb.value is True

    def test_missing_keyword(self):
        fb = ContainsKeywords(keywords=["hello", "world"])(outputs="hello there")
        assert fb.value is False
        assert "world" in fb.rationale

    def test_case_insensitive(self):
        fb = ContainsKeywords(keywords=["Hello"])(outputs="HELLO world")
        assert fb.value is True

    def test_case_sensitive(self):
        fb = ContainsKeywords(keywords=["Hello"], case_sensitive=True)(outputs="HELLO world")
        assert fb.value is False

    def test_phrase_keyword(self):
        fb = ContainsKeywords(keywords=["not financial advice"])(
            outputs="This is not financial advice."
        )
        assert fb.value is True

    def test_none_outputs(self):
        fb = ContainsKeywords(keywords=["a"])(outputs=None)
        assert fb.value is False


# ── LengthBound ─────────────────────────────────────────────────────────────


class TestLengthBound:
    def test_within_char_bounds(self):
        fb = LengthBound(min_length=1, max_length=100)(outputs="hello")
        assert fb.value is True

    def test_below_min_chars(self):
        fb = LengthBound(min_length=10)(outputs="hi")
        assert fb.value is False
        assert "below minimum" in fb.rationale

    def test_above_max_chars(self):
        fb = LengthBound(max_length=5)(outputs="hello world")
        assert fb.value is False
        assert "above maximum" in fb.rationale

    def test_word_count(self):
        fb = LengthBound(min_length=2, max_length=5, unit="words")(outputs="hello beautiful world")
        assert fb.value is True

    def test_word_count_below(self):
        fb = LengthBound(min_length=3, unit="words")(outputs="one two")
        assert fb.value is False

    def test_no_bounds(self):
        fb = LengthBound()(outputs="anything")
        assert fb.value is True

    def test_none_outputs(self):
        fb = LengthBound(min_length=1)(outputs=None)
        assert fb.value is False


# ── IsNotEmpty ──────────────────────────────────────────────────────────────


class TestIsNotEmpty:
    def test_non_empty(self):
        fb = IsNotEmpty()(outputs="hello")
        assert fb.value is True

    def test_empty_string(self):
        fb = IsNotEmpty()(outputs="")
        assert fb.value is False

    def test_whitespace_only(self):
        fb = IsNotEmpty()(outputs="   \t\n  ")
        assert fb.value is False

    def test_none_outputs(self):
        fb = IsNotEmpty()(outputs=None)
        assert fb.value is False

    def test_non_string(self):
        fb = IsNotEmpty()(outputs=42)
        assert fb.value is True


# ── LatencyThreshold ────────────────────────────────────────────────────────


class TestLatencyThreshold:
    def _make_trace(self, duration_ms):
        trace = MagicMock()
        trace.info.execution_duration = duration_ms
        return trace

    def test_within_threshold(self):
        fb = LatencyThreshold(max_latency_seconds=3.0)(trace=self._make_trace(2000))
        assert fb.value is True
        assert "within threshold" in fb.rationale

    def test_exceeds_threshold(self):
        fb = LatencyThreshold(max_latency_seconds=1.0)(trace=self._make_trace(2000))
        assert fb.value is False
        assert "above threshold" in fb.rationale

    def test_exact_threshold(self):
        fb = LatencyThreshold(max_latency_seconds=2.0)(trace=self._make_trace(2000))
        assert fb.value is True

    def test_no_trace(self):
        fb = LatencyThreshold(max_latency_seconds=1.0)(trace=None)
        assert fb.value is False

    def test_no_duration(self):
        trace = MagicMock()
        trace.info.execution_duration = None
        fb = LatencyThreshold(max_latency_seconds=1.0)(trace=trace)
        assert fb.value is False


# ── NumericBound ────────────────────────────────────────────────────────────


class TestNumericBound:
    def test_within_bounds(self):
        fb = NumericBound(min_value=0.0, max_value=1.0)(outputs=0.5)
        assert fb.value is True

    def test_below_min(self):
        fb = NumericBound(min_value=0.0)(outputs=-1)
        assert fb.value is False
        assert "below minimum" in fb.rationale

    def test_above_max(self):
        fb = NumericBound(max_value=100)(outputs=150)
        assert fb.value is False
        assert "above maximum" in fb.rationale

    def test_string_numeric(self):
        fb = NumericBound(min_value=0)(outputs="42")
        assert fb.value is True

    def test_non_numeric(self):
        fb = NumericBound()(outputs="not a number")
        assert fb.value is False
        assert "cannot be converted" in fb.rationale

    def test_none_outputs(self):
        fb = NumericBound()(outputs=None)
        assert fb.value is False

    def test_boundary_values(self):
        scorer = NumericBound(min_value=0.0, max_value=1.0)
        assert scorer(outputs=0.0).value is True
        assert scorer(outputs=1.0).value is True

    def test_no_bounds(self):
        fb = NumericBound()(outputs=999999)
        assert fb.value is True

    def test_nan_rejected(self):
        fb = NumericBound(min_value=0, max_value=1)(outputs=float("nan"))
        assert fb.value is False
        assert "not a finite number" in fb.rationale

    def test_inf_rejected(self):
        fb = NumericBound()(outputs=float("inf"))
        assert fb.value is False
        assert "not a finite number" in fb.rationale

    def test_negative_inf_rejected(self):
        fb = NumericBound()(outputs=float("-inf"))
        assert fb.value is False
        assert "not a finite number" in fb.rationale


# ── PII ─────────────────────────────────────────────────────────────────────


class TestPII:
    def test_no_pii(self):
        fb = PII()(outputs="Hello, how can I help you?")
        assert fb.value is True
        assert "No PII" in fb.rationale

    def test_email_detected(self):
        fb = PII()(outputs="Contact me at user@example.com")
        assert fb.value is False
        assert "email" in fb.rationale

    def test_phone_detected(self):
        fb = PII()(outputs="Call me at 555-123-4567")
        assert fb.value is False
        assert "phone" in fb.rationale

    def test_ssn_detected(self):
        fb = PII()(outputs="My SSN is 123-45-6789")
        assert fb.value is False
        assert "ssn" in fb.rationale

    def test_filter_pii_types(self):
        fb = PII(pii_types=["email"])(outputs="Call 555-123-4567")
        assert fb.value is True  # phone not checked

    def test_filter_pii_types_match(self):
        fb = PII(pii_types=["email"])(outputs="Email: user@example.com")
        assert fb.value is False

    def test_none_outputs(self):
        fb = PII()(outputs=None)
        assert fb.value is True

    def test_multiple_pii(self):
        fb = PII()(outputs="Contact user@a.com or user@b.com, call 555-123-4567")
        assert fb.value is False
        assert "email" in fb.rationale
        assert "phone" in fb.rationale

    def test_credit_card_detected(self):
        # Visa test number (passes Luhn)
        fb = PII()(outputs="My card is 4111111111111111")
        assert fb.value is False
        assert "credit_card" in fb.rationale

    def test_credit_card_luhn_rejects_random_digits(self):
        # 16 random digits that fail Luhn should not be flagged as credit card
        fb = PII(pii_types=["credit_card"])(outputs="Ref: 1234567890123456")
        assert fb.value is True


# ── Serialization ───────────────────────────────────────────────────────────


class TestBuiltInCodeScorerSerialization:
    def test_round_trip(self):
        scorer = JsonValidity(required_keys=["answer"])
        dumped = scorer.model_dump()
        assert dumped["builtin_scorer_class"] == "JsonValidity"
        assert dumped["builtin_scorer_pydantic_data"]["required_keys"] == ["answer"]

        from mlflow.genai.scorers.builtin_code_scorers import BuiltInCodeScorer

        restored = BuiltInCodeScorer.model_validate(dumped)
        assert isinstance(restored, JsonValidity)
        assert restored.required_keys == ["answer"]

    def test_round_trip_via_scorer_base(self):
        from mlflow.genai.scorers.base import Scorer

        scorer = LengthBound(min_length=10, max_length=200, unit="words")
        dumped = scorer.model_dump()
        restored = Scorer.model_validate(dumped)
        assert isinstance(restored, LengthBound)
        assert restored.min_length == 10
        assert restored.max_length == 200
        assert restored.unit == "words"

    def test_round_trip_regex_match(self):
        from mlflow.genai.scorers.base import Scorer

        scorer = RegexMatch(pattern=r"\d+", full_match=True)
        dumped = scorer.model_dump()
        restored = Scorer.model_validate(dumped)
        assert isinstance(restored, RegexMatch)
        assert restored.pattern == r"\d+"
        assert restored.full_match is True


# ── validate_columns ────────────────────────────────────────────────────────


class TestValidateColumns:
    def test_validate_columns_passes(self):
        scorer = ExactMatch()
        scorer.validate_columns({"outputs", "expectations"})

    def test_validate_columns_missing(self):
        scorer = ExactMatch()
        with pytest.raises(MlflowException, match="requires columns"):
            scorer.validate_columns({"outputs"})
