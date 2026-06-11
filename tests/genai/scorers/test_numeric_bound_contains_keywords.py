"""Tests for NumericBound and ContainsKeywords built-in deterministic scorers."""

import math

import pytest

from mlflow.genai.judges.utils import CategoricalRating

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yes(feedback):
    assert feedback.value == CategoricalRating.YES, (
        f"Expected YES, got {feedback.value!r}: {feedback.rationale}"
    )


def _no(feedback):
    assert feedback.value == CategoricalRating.NO, (
        f"Expected NO, got {feedback.value!r}: {feedback.rationale}"
    )


# ---------------------------------------------------------------------------
# NumericBound
# ---------------------------------------------------------------------------


class TestNumericBound:
    @pytest.fixture
    def scorer(self):
        from mlflow.genai.scorers import NumericBound

        return NumericBound(min_value=0.0, max_value=1.0)

    def test_value_within_bounds_passes(self, scorer):
        _yes(scorer(outputs=0.5))

    def test_value_at_min_inclusive_passes(self, scorer):
        _yes(scorer(outputs=0.0))

    def test_value_at_max_inclusive_passes(self, scorer):
        _yes(scorer(outputs=1.0))

    def test_value_below_min_fails(self, scorer):
        _no(scorer(outputs=-0.1))

    def test_value_above_max_fails(self, scorer):
        _no(scorer(outputs=1.1))

    def test_string_coercion_passes(self):
        from mlflow.genai.scorers import NumericBound

        _yes(NumericBound(min_value=0, max_value=100)(outputs="42"))

    def test_string_coercion_fails_non_numeric(self):
        from mlflow.genai.scorers import NumericBound

        _no(NumericBound(min_value=0, max_value=100)(outputs="hello"))

    def test_int_output(self):
        from mlflow.genai.scorers import NumericBound

        _yes(NumericBound(min_value=0, max_value=100)(outputs=50))

    def test_nan_fails(self, scorer):
        _no(scorer(outputs=float("nan")))

    def test_pos_inf_fails(self, scorer):
        _no(scorer(outputs=math.inf))

    def test_neg_inf_fails(self, scorer):
        _no(scorer(outputs=-math.inf))

    def test_bool_rejected(self, scorer):
        _no(scorer(outputs=True))

    def test_none_outputs_fails(self, scorer):
        _no(scorer(outputs=None))

    def test_field_extraction_dict(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0, field="confidence")
        _yes(scorer(outputs={"confidence": 0.87}))

    def test_field_extraction_missing_key(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0, field="confidence")
        _no(scorer(outputs={"score": 0.5}))

    def test_field_extraction_non_dict(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0, field="confidence")
        _no(scorer(outputs="not a dict"))

    def test_list_all_pass(self):
        from mlflow.genai.scorers import NumericBound

        _yes(NumericBound(min_value=0, max_value=10)(outputs=[1, 5, 10]))

    def test_list_one_fails(self):
        from mlflow.genai.scorers import NumericBound

        fb = NumericBound(min_value=0, max_value=10)(outputs=[1, 5, 15])
        _no(fb)
        assert "[2]" in fb.rationale

    def test_list_string_coercion(self):
        from mlflow.genai.scorers import NumericBound

        _yes(NumericBound(min_value=0, max_value=10)(outputs=["1", "5", "10"]))

    def test_exclusive_bounds_at_edge_fails(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0, inclusive=False)
        _no(scorer(outputs=0.0))
        _no(scorer(outputs=1.0))

    def test_exclusive_bounds_inside_passes(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=0.0, max_value=1.0, inclusive=False)
        _yes(scorer(outputs=0.5))

    def test_only_min_value(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(min_value=5.0)
        _yes(scorer(outputs=100.0))
        _no(scorer(outputs=4.9))

    def test_only_max_value(self):
        from mlflow.genai.scorers import NumericBound

        scorer = NumericBound(max_value=10.0)
        _yes(scorer(outputs=-999.0))
        _no(scorer(outputs=10.1))

    def test_validation_no_bounds_raises(self):
        from mlflow.genai.scorers import NumericBound

        with pytest.raises(ValueError, match="at least one"):
            NumericBound()

    def test_validation_min_gt_max_raises(self):
        from mlflow.genai.scorers import NumericBound

        with pytest.raises(ValueError, match="must be <="):
            NumericBound(min_value=10.0, max_value=5.0)

    def test_validation_nan_min_raises(self):
        from mlflow.genai.scorers import NumericBound

        with pytest.raises(ValueError, match="must not be NaN"):
            NumericBound(min_value=float("nan"), max_value=1.0)

    def test_validation_nan_max_raises(self):
        from mlflow.genai.scorers import NumericBound

        with pytest.raises(ValueError, match="must not be NaN"):
            NumericBound(min_value=0.0, max_value=float("nan"))

    def test_inf_bound_leaves_side_unbounded(self):
        from mlflow.genai.scorers import NumericBound

        # +inf max bound is allowed and leaves the upper side effectively unbounded,
        # but a +inf *output* still fails as a degenerate value.
        scorer = NumericBound(min_value=0.0, max_value=math.inf)
        _yes(scorer(outputs=1e308))
        _no(scorer(outputs=math.inf))

    def test_rationale_contains_value(self, scorer):
        fb = scorer(outputs=0.5)
        assert "0.5" in fb.rationale

    def test_rationale_on_failure_mentions_bound(self, scorer):
        fb = scorer(outputs=2.0)
        assert "maximum" in fb.rationale or "1.0" in fb.rationale


# ---------------------------------------------------------------------------
# ContainsKeywords
# ---------------------------------------------------------------------------


class TestContainsKeywords:
    @pytest.fixture
    def scorer(self):
        from mlflow.genai.scorers import ContainsKeywords

        return ContainsKeywords(keywords=["disclaimer", "not financial advice"])

    def test_all_keywords_present_passes(self, scorer):
        _yes(scorer(outputs="This is not financial advice. Please add a disclaimer."))

    def test_missing_one_keyword_fails(self, scorer):
        fb = scorer(outputs="This is not financial advice.")
        _no(fb)
        assert "disclaimer" in fb.rationale

    def test_missing_all_keywords_fails(self, scorer):
        _no(scorer(outputs="Hello world"))

    def test_case_insensitive_by_default(self, scorer):
        _yes(scorer(outputs="NOT FINANCIAL ADVICE and DISCLAIMER here"))

    def test_case_sensitive_mode(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["Disclaimer"], case_sensitive=True)
        _no(scorer(outputs="disclaimer present"))
        _yes(scorer(outputs="Disclaimer present"))

    def test_mode_any_one_found_passes(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["yes", "confirmed", "approved"], mode="any")
        _yes(scorer(outputs="Your request has been approved."))

    def test_mode_any_none_found_fails(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["yes", "confirmed", "approved"], mode="any")
        _no(scorer(outputs="I cannot do that."))

    def test_whole_word_avoids_substring(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["not"], whole_word=True)
        _no(scorer(outputs="nothing to see here"))
        _yes(scorer(outputs="I am not sure"))

    def test_substring_match_by_default(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["not"])
        _yes(scorer(outputs="nothing to see here"))

    def test_none_outputs_fails(self, scorer):
        _no(scorer(outputs=None))

    def test_empty_string_output_fails(self, scorer):
        _no(scorer(outputs=""))

    def test_dict_output_converted_to_str(self, scorer):
        _yes(scorer(outputs={"text": "not financial advice and disclaimer"}))

    def test_validation_empty_keywords_raises(self):
        from mlflow.genai.scorers import ContainsKeywords

        with pytest.raises(Exception, match="non-empty"):
            ContainsKeywords(keywords=[])

    def test_validation_empty_string_keyword_raises(self):
        from mlflow.genai.scorers import ContainsKeywords

        with pytest.raises(Exception, match="empty strings"):
            ContainsKeywords(keywords=["valid", ""])

    def test_rationale_lists_missing_on_failure(self, scorer):
        fb = scorer(outputs="disclaimer only")
        assert "not financial advice" in fb.rationale

    def test_rationale_confirms_found_on_pass(self, scorer):
        fb = scorer(outputs="This is not financial advice. A disclaimer is included.")
        assert "found" in fb.rationale

    def test_multi_word_phrase_matching(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["not financial advice"])
        _yes(scorer(outputs="Remember: this is not financial advice."))
        _no(scorer(outputs="This is financial advice."))

    def test_single_keyword(self):
        from mlflow.genai.scorers import ContainsKeywords

        scorer = ContainsKeywords(keywords=["hello"])
        _yes(scorer(outputs="hello world"))
        _no(scorer(outputs="goodbye world"))

    def test_instructions_reflect_whole_word_setting(self):
        from mlflow.genai.scorers import ContainsKeywords

        assert "substring" in ContainsKeywords(keywords=["x"]).instructions
        assert "whole-word" in ContainsKeywords(keywords=["x"], whole_word=True).instructions
