import json
from typing import Literal

import pandas as pd
import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import Scorer, ScorerKind, _extract_scorer_value, scorer_ensemble
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


def test_majority_vote_picks_most_common():
    fb = majority_vote([True, True, False])
    assert isinstance(fb, Feedback)
    assert fb.value is True


def test_majority_vote_breaks_ties_lexicographically():
    # Tie between True and False -> lexicographic order of str(value): "False" < "True"
    assert majority_vote([True, False]).value is False
    assert majority_vote(["b", "a"]).value == "a"


def test_mean_averages():
    assert mean([1.0, 2.0, 3.0]).value == 2.0


def test_minimum_maximum():
    assert minimum([3, 1, 2]).value == 1
    assert maximum([3, 1, 2]).value == 3


def test_agg_all_and_any():
    assert agg_all([True, True]).value is True
    assert agg_all([True, False]).value is False
    assert agg_any([False, False]).value is False
    assert agg_any([False, True]).value is True


@pytest.mark.parametrize("fn", [majority_vote, mean, minimum, maximum, agg_all, agg_any])
def test_builtins_raise_on_none_entry(fn):
    with pytest.raises(MlflowException, match="failed"):
        fn([True, None])


@pytest.mark.parametrize("fn", [majority_vote, mean, minimum, maximum, agg_all, agg_any])
def test_builtins_raise_on_empty_input(fn):
    with pytest.raises(MlflowException, match="failed"):
        fn([])


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


def test_scorer_kind_has_ensemble():
    assert ScorerKind.ENSEMBLE.value == "ensemble"


def test_extract_value_from_feedback():
    assert _extract_scorer_value(Feedback(value=0.5)) == 0.5


def test_extract_value_from_primitive():
    assert _extract_scorer_value(True) is True
    assert _extract_scorer_value(3) == 3


def test_extract_value_none_for_none_and_error():
    assert _extract_scorer_value(None) is None
    assert _extract_scorer_value(Feedback(error=ValueError("boom"))) is None


def test_extract_value_rejects_feedback_list():
    with pytest.raises(MlflowException, match="list"):
        _extract_scorer_value([Feedback(value=1), Feedback(value=2)])


@scorer
def _always_true(outputs) -> bool:
    return True


@scorer
def _always_false(outputs) -> bool:
    return False


@scorer
def _num_len(outputs) -> int:
    return len(outputs)


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


# --- Serialization tests ---
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


def test_ensemble_custom_callable_not_serializable():
    agg = scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: sum(values))
    with pytest.raises(MlflowException, match="custom ensemble function"):
        agg.model_dump()


def test_public_exports():
    import mlflow.genai
    from mlflow.genai.scorers import (
        agg_all,
        agg_any,
        majority_vote,
        maximum,
        mean,
        minimum,
        scorer_ensemble,
    )

    assert mlflow.genai.scorer_ensemble is scorer_ensemble
    assert all(callable(fn) for fn in (majority_vote, mean, minimum, maximum, agg_all, agg_any))


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
# Task 7: categorical/Literal values + feedback_value_type validation
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
    # feedbacks-mode bypasses the value-type check entirely; feedbacks contains raw run() results.
    def first_value(feedbacks):
        return feedbacks[0]

    agg = scorer_ensemble(name="f", scorers=[_label], ensemble_fn=first_value)
    assert agg(outputs="hello").value == "hello"


def test_numeric_builtin_rejects_literal_judge_upfront():
    # Safety declares feedback_value_type == Literal["yes","no"]; mean must reject it early.
    with pytest.raises(MlflowException, match="numeric"):
        scorer_ensemble(name="x", scorers=[Safety()], ensemble_fn="mean")


def test_majority_vote_accepts_literal_judge():
    # majority_vote is categorical-friendly; Safety() constructs fine.
    agg = scorer_ensemble(name="ok", scorers=[Safety()], ensemble_fn="majority_vote")
    assert agg.name == "ok"


def test_is_numeric_feedback_type_unit_checks():
    assert is_numeric_feedback_type(float) is True
    assert is_numeric_feedback_type(int) is True
    assert is_numeric_feedback_type(bool) is True
    assert is_numeric_feedback_type(Literal[1, 2, 3]) is True
    assert is_numeric_feedback_type(Literal["yes", "no"]) is False
    assert is_numeric_feedback_type(str) is False
    assert is_numeric_feedback_type(None) is False


# ---------------------------------------------------------------------------
# Task 10: sub-scorer provenance preserved in ensemble Feedback.metadata
# ---------------------------------------------------------------------------


def test_ensemble_preserves_sub_rationales_in_metadata():
    from mlflow.genai.scorers.base import EnsembleScorer

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
    assert {e["rationale"] for e in sub} == {"looks safe", "found an issue"}
    assert {e["name"] for e in sub} == {"_yes", "_no"}


def test_ensemble_metadata_present_for_custom_value_fn():
    from mlflow.genai.scorers.base import EnsembleScorer

    agg = scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: max(values))
    fb = agg(outputs="abcd")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert len(sub) == 1
    assert sub[0]["name"] == "_num_len"
    assert sub[0]["value"] == 4


def test_ensemble_fn_metadata_wins_on_collision():
    from mlflow.genai.scorers.base import EnsembleScorer

    def fn_with_meta(feedbacks):
        return Feedback(value=True, metadata={"custom": "kept"})

    agg = scorer_ensemble(name="m", scorers=[_always_true], ensemble_fn=fn_with_meta)
    fb = agg(outputs="x")
    assert fb.metadata["custom"] == "kept"
    # sub-feedback provenance is still attached alongside it
    assert EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY in fb.metadata


def test_ensemble_metadata_is_string_valued():
    agg = scorer_ensemble(name="s", scorers=[_always_true], ensemble_fn="agg_all")
    fb = agg(outputs="x")
    for v in fb.metadata.values():
        assert isinstance(v, str)
