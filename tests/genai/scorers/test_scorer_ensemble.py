import json
from typing import Literal

import pandas as pd
import pytest

import mlflow.genai
import mlflow.genai.scorers as _public_scorers
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import (
    EnsembleScorer,
    Scorer,
    ScorerKind,
    _extract_scorer_value,
    scorer_ensemble,
)
from mlflow.genai.scorers.builtin_scorers import Safety
from mlflow.genai.scorers.ensemble import (
    BUILTIN_ENSEMBLES,
    agg_all,
    agg_any,
    is_numeric_feedback_type,
    majority_vote,
    maximum,
    mean,
    minimum,
)

# ---------------------------------------------------------------------------
# Built-in ensemble function unit tests
# ---------------------------------------------------------------------------

_ALL_BUILTINS = [majority_vote, mean, minimum, maximum, agg_all, agg_any]
_NUMERIC_BUILTINS = [mean, minimum, maximum]


@pytest.mark.parametrize(
    ("fn", "values", "expected"),
    [
        # majority_vote
        pytest.param(majority_vote, [True, True, False], True, id="majority_vote-basic"),
        pytest.param(majority_vote, [True, False], False, id="majority_vote-tie_false_wins"),
        pytest.param(majority_vote, ["b", "a"], "a", id="majority_vote-categorical_tie"),
        pytest.param(majority_vote, ["x", "x", "y"], "x", id="majority_vote-categorical_majority"),
        pytest.param(majority_vote, [True], True, id="majority_vote-single"),
        pytest.param(majority_vote, ["only"], "only", id="majority_vote-single_categorical"),
        # mean
        pytest.param(mean, [1.0, 2.0, 3.0], 2.0, id="mean-basic"),
        pytest.param(mean, [2], 2, id="mean-single"),
        pytest.param(mean, [True, False], 0.5, id="mean-bools"),
        pytest.param(mean, [-1.0, 1.0], 0.0, id="mean-zero"),
        # minimum
        pytest.param(minimum, [3, 1, 2], 1, id="minimum-basic"),
        pytest.param(minimum, [5], 5, id="minimum-single"),
        pytest.param(minimum, [True, False], False, id="minimum-bools"),
        # maximum
        pytest.param(maximum, [3, 1, 2], 3, id="maximum-basic"),
        pytest.param(maximum, [5], 5, id="maximum-single"),
        pytest.param(maximum, [True, False], True, id="maximum-bools"),
        # agg_all
        pytest.param(agg_all, [True, True], True, id="agg_all-all_true"),
        pytest.param(agg_all, [True, False], False, id="agg_all-mixed"),
        pytest.param(agg_all, [True], True, id="agg_all-single"),
        # agg_any
        pytest.param(agg_any, [False, False], False, id="agg_any-all_false"),
        pytest.param(agg_any, [False, True], True, id="agg_any-mixed"),
        pytest.param(agg_any, [False], False, id="agg_any-single"),
    ],
)
def test_builtin_happy_path(fn, values, expected):
    result = fn(values)
    assert isinstance(result, Feedback)
    if isinstance(expected, bool):
        assert result.value is expected
    else:
        assert result.value == expected


@pytest.mark.parametrize("fn", _ALL_BUILTINS)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([True, None], id="with_none"),
        pytest.param([None], id="all_none"),
    ],
)
def test_builtins_raise_on_none_entry(fn, values):
    with pytest.raises(MlflowException, match="failed"):
        fn(values)


@pytest.mark.parametrize("fn", _ALL_BUILTINS)
def test_builtins_raise_on_empty_input(fn):
    with pytest.raises(MlflowException, match="failed"):
        fn([])


@pytest.mark.parametrize("fn", _NUMERIC_BUILTINS)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(["a", "b"], id="all_strings"),
        pytest.param([1.0, "b"], id="mixed"),
    ],
)
def test_numeric_builtins_reject_non_numeric(fn, values):
    with pytest.raises(MlflowException, match="numeric"):
        fn(values)


def test_builtin_registry_maps_names():
    assert BUILTIN_ENSEMBLES["majority_vote"] is majority_vote
    assert BUILTIN_ENSEMBLES["mean"] is mean
    assert set(BUILTIN_ENSEMBLES) == {
        "majority_vote",
        "mean",
        "minimum",
        "maximum",
        "agg_all",
        "agg_any",
    }


# ---------------------------------------------------------------------------
# ScorerKind and _extract_scorer_value unit tests
# ---------------------------------------------------------------------------


def test_scorer_kind_has_ensemble():
    assert ScorerKind.ENSEMBLE.value == "ensemble"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(Feedback(value=0.5), 0.5, id="feedback_value"),
        pytest.param(True, True, id="bool_primitive"),
        pytest.param(3, 3, id="int_primitive"),
        pytest.param(None, None, id="none_input"),
        pytest.param(Feedback(error=ValueError("boom")), None, id="error_feedback"),
    ],
)
def test_extract_scorer_value(value, expected):
    result = _extract_scorer_value(value)
    if isinstance(expected, bool):
        assert result is expected
    else:
        assert result == expected


def test_extract_value_rejects_feedback_list():
    with pytest.raises(MlflowException, match="list"):
        _extract_scorer_value([Feedback(value=1), Feedback(value=2)])


# ---------------------------------------------------------------------------
# Module-level scorer fixtures used across integration tests
# ---------------------------------------------------------------------------


@scorer
def _always_true(outputs) -> bool:
    return True


@scorer
def _always_false(outputs) -> bool:
    return False


@scorer
def _num_len(outputs) -> int:
    return len(outputs)


# ---------------------------------------------------------------------------
# EnsembleScorer integration tests
# ---------------------------------------------------------------------------


def test_ensemble_majority_vote_end_to_end():
    agg = scorer_ensemble(
        name="vote",
        scorers=[_always_true, _always_true, _always_false],
        ensemble_fn="majority_vote",
    )
    fb = agg(outputs="hello")
    assert fb.value is True
    assert fb.name == "vote"


def test_ensemble_dispatches_mixed_signatures():
    # _always_true takes outputs; _num_len takes outputs too; use maximum over ints/bools.
    agg = scorer_ensemble(name="max_agg", scorers=[_num_len], ensemble_fn="maximum")
    assert agg(outputs="abcd").value == 4


def test_ensemble_custom_callable_on_values():
    def spread(values):
        return max(values) - min(values)

    agg = scorer_ensemble(name="spread", scorers=[_num_len, _num_len], ensemble_fn=spread)
    fb = agg(outputs="abc")
    assert fb.value == 0  # both return 3
    assert fb.name == "spread"


def test_ensemble_feedbacks_mode():
    def count_feedbacks(feedbacks):
        return len(feedbacks)

    agg = scorer_ensemble(
        name="count", scorers=[_always_true, _always_false], ensemble_fn=count_feedbacks
    )
    assert agg(outputs="x").value == 2


def test_ensemble_rejects_unsupported_value_type():
    # A scorer returning a non-str/numeric type (dict) is still rejected in values-mode.
    @scorer
    def _returns_dict(outputs) -> dict[str, str]:
        return {"key": "val"}

    agg = scorer_ensemble(name="bad", scorers=[_returns_dict], ensemble_fn="majority_vote")
    with pytest.raises(MlflowException, match="_returns_dict"):
        agg(outputs="x")


def test_ensemble_rejects_empty_scorers():
    with pytest.raises(MlflowException, match="at least one"):
        scorer_ensemble(name="e", scorers=[], ensemble_fn="mean")


def test_ensemble_rejects_unknown_builtin_name():
    with pytest.raises(MlflowException, match="not a built-in"):
        scorer_ensemble(name="e", scorers=[_always_true], ensemble_fn="bogus")


def test_ensemble_session_level_derivation_and_mixing():
    @scorer
    def _sess(session) -> bool:
        return len(session) > 0

    session_agg = scorer_ensemble(name="s", scorers=[_sess], ensemble_fn="agg_all")
    assert session_agg.is_session_level_scorer is True

    single_agg = scorer_ensemble(name="t", scorers=[_always_true], ensemble_fn="agg_all")
    assert single_agg.is_session_level_scorer is False

    with pytest.raises(MlflowException, match="same level"):
        scorer_ensemble(name="m", scorers=[_sess, _always_true], ensemble_fn="agg_all")


def test_ensemble_records_builtin_name_for_callable():
    agg = scorer_ensemble(name="v", scorers=[_always_true], ensemble_fn=majority_vote)
    assert agg._ensemble_fn_name == "majority_vote"


def test_ensemble_custom_callable_has_no_builtin_name():
    agg = scorer_ensemble(name="c", scorers=[_always_true], ensemble_fn=lambda values: max(values))
    assert agg._ensemble_fn_name is None


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------
# Round-trip tests use Safety() (a builtin scorer) as the sub-scorer because decorator-backed
# sub-scorers require a Databricks tracking URI to reconstruct (_reconstruct_decorator_scorer
# raises outside Databricks). Safety serializes/reconstructs locally without any network call.


def test_ensemble_serialization_produces_ensemble_scorer_data():
    """model_dump() on an ensemble scorer must produce ensemble_scorer_data, not trip the
    SerializedScorer mutual-exclusion validator.
    """
    # Safety declares Literal["yes","no"]; use majority_vote (categorical-friendly).
    agg = scorer_ensemble(
        name="agg_safety",
        scorers=[Safety()],
        ensemble_fn="majority_vote",
    )
    dumped = agg.model_dump()
    assert "ensemble_scorer_data" in dumped
    assert dumped["ensemble_scorer_data"]["ensemble_fn"] == "majority_vote"
    assert len(dumped["ensemble_scorer_data"]["scorers"]) == 1


def test_ensemble_serialization_round_trip():
    """Full dump → validate round-trip using a builtin sub-scorer (Safety) to avoid the
    OSS decorator-reconstruction block.
    """
    # Safety declares Literal["yes","no"]; use majority_vote (categorical-friendly).
    agg = scorer_ensemble(
        name="agg_safety",
        scorers=[Safety()],
        ensemble_fn="majority_vote",
    )
    dumped = agg.model_dump()
    assert dumped["ensemble_scorer_data"]["ensemble_fn"] == "majority_vote"
    assert len(dumped["ensemble_scorer_data"]["scorers"]) == 1

    restored = Scorer.model_validate(dumped)
    assert restored.name == "agg_safety"
    assert restored.kind.value == "ensemble"
    assert restored._ensemble_fn_name == "majority_vote"
    assert len(restored._scorers) == 1
    assert isinstance(restored._scorers[0], Safety)


def test_ensemble_serialization_preserves_aggregations():
    agg = scorer_ensemble(
        name="agg_safety",
        scorers=[Safety()],
        ensemble_fn="majority_vote",
        aggregations=["min", "max"],
    )
    dumped = agg.model_dump()
    assert dumped["aggregations"] == ["min", "max"]
    restored = Scorer.model_validate(dumped)
    assert restored.aggregations == ["min", "max"]


def test_ensemble_custom_callable_not_serializable():
    agg = scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: sum(values))
    with pytest.raises(MlflowException, match="custom ensemble function"):
        agg.model_dump()


def test_public_exports():
    assert mlflow.genai.scorer_ensemble is scorer_ensemble
    for name, fn in [
        ("majority_vote", majority_vote),
        ("mean", mean),
        ("minimum", minimum),
        ("maximum", maximum),
        ("agg_all", agg_all),
        ("agg_any", agg_any),
        ("scorer_ensemble", scorer_ensemble),
    ]:
        assert getattr(_public_scorers, name) is fn


def test_ensemble_in_evaluate(is_in_databricks):
    data = pd.DataFrame({
        "inputs": [{"q": "a"}, {"q": "b"}],
        "outputs": ["yes", "no"],
    })
    agg = scorer_ensemble(
        name="ensemble",
        scorers=[_always_true, _always_false],
        ensemble_fn="agg_any",
    )
    result = mlflow.genai.evaluate(data=data, scorers=[agg])
    # agg_any is order-independent, so the ensemble feedback is logged for every row
    assert "ensemble/mean" in result.metrics


# ---------------------------------------------------------------------------
# Categorical / Literal values and feedback_value_type validation
# ---------------------------------------------------------------------------


@scorer
def _label(outputs) -> str:
    return outputs


def test_ensemble_majority_vote_categorical():
    # majority_vote now handles categorical string values.
    agg = scorer_ensemble(name="cat", scorers=[_label, _label, _label], ensemble_fn="majority_vote")
    assert agg(outputs="yes").value == "yes"


def test_mean_rejects_categorical_values():
    # mean must raise for string values at call time, not silently crash in statistics.mean.
    agg = scorer_ensemble(name="m", scorers=[_label], ensemble_fn="mean")
    with pytest.raises(MlflowException, match="numeric"):
        agg(outputs="0.5")


def test_feedbacks_mode_allows_non_numeric():
    # feedbacks-mode bypasses the value-type check entirely; feedbacks contains Feedback objects.
    def first_value(feedbacks):
        return feedbacks[0].value

    agg = scorer_ensemble(name="f", scorers=[_label], ensemble_fn=first_value)
    assert agg(outputs="hello").value == "hello"


def test_ensemble_feedbacks_mode_receives_feedback_objects():
    captured = {}

    def inspect_fn(feedbacks):
        captured["types"] = [type(f).__name__ for f in feedbacks]
        captured["names"] = [f.name for f in feedbacks]
        return feedbacks[0].value

    # _num_len returns a bare int; must be wrapped into a Feedback in feedbacks-mode.
    agg = scorer_ensemble(name="fb", scorers=[_num_len], ensemble_fn=inspect_fn)
    fb = agg(outputs="abcd")
    assert captured["types"] == ["Feedback"]
    assert captured["names"] == ["_num_len"]
    assert fb.value == 4


def test_numeric_builtin_rejects_literal_judge_upfront():
    # Safety declares feedback_value_type == Literal["yes","no"]; mean must reject it early.
    with pytest.raises(MlflowException, match="numeric"):
        scorer_ensemble(name="x", scorers=[Safety()], ensemble_fn="mean")


def test_majority_vote_accepts_literal_judge():
    # majority_vote is categorical-friendly; Safety() constructs fine.
    agg = scorer_ensemble(name="ok", scorers=[Safety()], ensemble_fn="majority_vote")
    assert agg.name == "ok"


@pytest.mark.parametrize(
    ("feedback_type", "expected"),
    [
        pytest.param(float, True, id="float"),
        pytest.param(int, True, id="int"),
        pytest.param(bool, True, id="bool"),
        pytest.param(Literal[1, 2, 3], True, id="literal_numeric"),
        pytest.param(Literal["yes", "no"], False, id="literal_str"),
        pytest.param(str, False, id="str"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_numeric_feedback_type(feedback_type, expected):
    assert is_numeric_feedback_type(feedback_type) is expected


# ---------------------------------------------------------------------------
# Sub-scorer provenance in ensemble Feedback.metadata
# ---------------------------------------------------------------------------


def test_ensemble_preserves_sub_feedbacks_in_metadata():
    @scorer
    def _yes(outputs) -> Feedback:
        return Feedback(value=True, rationale="looks safe")

    @scorer
    def _no(outputs) -> Feedback:
        return Feedback(value=False, rationale="found an issue")

    agg = scorer_ensemble(name="e", scorers=[_yes, _no], ensemble_fn="majority_vote")
    fb = agg(outputs="x")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert len(sub) == 2
    assert {e["assessment_name"] for e in sub} == {"_yes", "_no"}
    assert {e["rationale"] for e in sub} == {"looks safe", "found an issue"}
    # values live under the nested "feedback" key
    assert {e["feedback"]["value"] for e in sub} == {True, False}


def test_ensemble_metadata_captures_sub_feedback_source():
    @scorer
    def _judge(outputs) -> Feedback:
        return Feedback(
            value=True,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
            ),
        )

    agg = scorer_ensemble(name="e", scorers=[_judge], ensemble_fn="agg_all")
    fb = agg(outputs="x")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["source"]["source_type"] == "LLM_JUDGE"
    assert sub[0]["source"]["source_id"] == "gpt-4o-mini"


def test_ensemble_metadata_normalizes_bare_value():
    agg = scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: max(values))
    fb = agg(outputs="abcd")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["assessment_name"] == "_num_len"
    assert sub[0]["feedback"]["value"] == 4
    assert sub[0]["source"]["source_type"] == "CODE"


def test_ensemble_fn_metadata_wins_on_collision():
    key = EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY

    def fn_with_meta(feedbacks):
        return Feedback(value=True, metadata={key: "override"})

    agg = scorer_ensemble(name="m", scorers=[_always_true], ensemble_fn=fn_with_meta)
    fb = agg(outputs="x")
    # The ensemble_fn's own metadata wins on key collision.
    assert fb.metadata[key] == "override"


def test_ensemble_metadata_is_string_valued():
    agg = scorer_ensemble(name="s", scorers=[_always_true], ensemble_fn="agg_all")
    fb = agg(outputs="x")
    for v in fb.metadata.values():
        assert isinstance(v, str)


def test_ensemble_metadata_captures_error_feedback():
    @scorer
    def _errs(outputs) -> Feedback:
        return Feedback(error=ValueError("boom"))

    def tolerant(feedbacks):
        return True

    agg = scorer_ensemble(name="e", scorers=[_errs], ensemble_fn=tolerant)
    fb = agg(outputs="x")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["assessment_name"] == "_errs"
    # to_dictionary() puts error info under the nested "feedback.error" key
    assert sub[0]["feedback"]["error"]["error_code"] == "ValueError"
    assert sub[0]["feedback"]["error"]["error_message"] == "boom"
