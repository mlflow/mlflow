import json
from typing import Literal
from unittest import mock

import pandas as pd
import pytest

import mlflow.genai
import mlflow.genai.scorers as _public_scorers
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import (
    EnsembleScorer,
    Scorer,
    ScorerKind,
    _extract_scorer_value,
    flatten_scorers,
    make_scorer_ensemble,
)
from mlflow.genai.scorers.builtin_scorers import Correctness, Safety
from mlflow.genai.scorers.ensemble import (
    BUILTIN_ENSEMBLES,
    agg_all,
    agg_any,
    is_bool_feedback_type,
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


@pytest.mark.parametrize("fn", [agg_all, agg_any])
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(["maybe", "no"], id="unmappable_string"),
        pytest.param([1, 2], id="ints_not_bool"),
        pytest.param([True, 1], id="mixed_bool_int"),
    ],
)
def test_bool_builtins_reject_values_without_yes_no_reading(fn, values):
    with pytest.raises(MlflowException, match="yes/no reading"):
        fn(values)


@pytest.mark.parametrize(
    ("fn", "values", "expected"),
    [
        # Built-in judges return Literal["yes","no"], not bools. Truthiness would make every
        # non-empty string True, so "no" must read as False.
        pytest.param(agg_all, ["yes", "no"], False, id="agg_all-yes_no"),
        pytest.param(agg_all, ["yes", "yes"], True, id="agg_all-all_yes"),
        pytest.param(agg_any, ["no", "no"], False, id="agg_any-all_no"),
        pytest.param(agg_any, ["no", "yes"], True, id="agg_any-one_yes"),
        pytest.param(agg_all, [CategoricalRating.YES, "yes"], True, id="agg_all-rating_enum"),
        pytest.param(agg_all, ["YES", " yes "], True, id="agg_all-case_and_space"),
        pytest.param(agg_all, ["pass", True], True, id="agg_all-affirmative-synonym"),
    ],
)
def test_bool_builtins_coerce_categorical_yes_no(fn, values, expected):
    result = fn(values)
    assert result.value is expected


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
    agg = make_scorer_ensemble(
        name="vote",
        scorers=[_always_true, _always_true, _always_false],
        ensemble_fn="majority_vote",
    )
    fb = agg(outputs="hello")
    assert fb.value is True
    assert fb.name == "vote"


def test_ensemble_dispatches_mixed_signatures():
    # _always_true takes outputs; _num_len takes outputs too; use maximum over ints/bools.
    agg = make_scorer_ensemble(name="max_agg", scorers=[_num_len], ensemble_fn="maximum")
    assert agg(outputs="abcd").value == 4


def test_ensemble_fn_c_builtin_no_signature():
    # C builtins like max expose no introspectable signature; the dispatch must not raise and
    # should default to values-mode (max over the extracted values).
    agg = make_scorer_ensemble(name="b", scorers=[_num_len, _num_len], ensemble_fn=max)
    assert agg(outputs="abcd").value == 4


def test_ensemble_resilient_to_sub_scorer_failure():
    @scorer
    def _boom(outputs) -> bool:
        raise ValueError("kaboom")

    def tolerant(feedbacks):
        # One sub-scorer crashed; a feedbacks-mode fn can still produce a result.
        return True

    agg = make_scorer_ensemble(name="e", scorers=[_always_true, _boom], ensemble_fn=tolerant)
    fb = agg(outputs="x")
    assert fb.value is True
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    crashed = next(e for e in sub if e["assessment_name"] == "_boom")
    assert crashed["feedback"]["error"]["error_code"] == "ValueError"


def test_builtin_ensemble_failure_names_the_crashed_sub_scorer():
    @scorer
    def _boom(outputs) -> bool:
        raise ValueError("kaboom")

    # Built-in fns treat a missing sub-scorer value as fatal. The raised error must still
    # identify which sub-scorer failed and why, rather than only "returned no value".
    agg = make_scorer_ensemble(name="e", scorers=[_always_true, _boom], ensemble_fn="agg_all")
    with pytest.raises(MlflowException, match="_boom.*kaboom"):
        agg(outputs="x")


@pytest.mark.parametrize(
    ("ensemble_fn", "expects_feedbacks"),
    [
        pytest.param(lambda values: values, False, id="values"),
        pytest.param(lambda feedbacks: feedbacks, True, id="feedbacks"),
        # The aggregate input is passed positionally, so a `feedbacks` keyword that is not
        # the first parameter must not flip the mode.
        pytest.param(lambda values, feedbacks=None: values, False, id="feedbacks-not-first"),
        pytest.param(lambda feedbacks, values=None: feedbacks, True, id="feedbacks-first"),
    ],
)
def test_dispatch_mode_follows_first_positional_parameter(ensemble_fn, expects_feedbacks):
    agg = make_scorer_ensemble(name="d", scorers=[_num_len], ensemble_fn=ensemble_fn)
    received = agg(outputs="abcd").value
    assert isinstance(received[0], Feedback) is expects_feedbacks


def test_ensemble_create_copy_recurses_into_sub_scorers():
    # Sub-scorers must be copied through their own _create_copy rather than deep-copied
    # wholesale: a deepcopy recurses infinitely on a third-party sub-scorer that holds an
    # `instructor`-wrapped client.
    sub = Safety()
    agg = make_scorer_ensemble(name="c", scorers=[sub], ensemble_fn="majority_vote")
    copy = agg._create_copy()
    assert isinstance(copy, EnsembleScorer)
    assert copy.name == "c"
    assert copy._ensemble_fn_name == "majority_vote"
    assert isinstance(copy._scorers[0], Safety)
    assert copy._scorers[0] is not sub


def test_ensemble_custom_callable_on_values():
    def spread(values):
        return max(values) - min(values)

    agg = make_scorer_ensemble(name="spread", scorers=[_num_len, _num_len], ensemble_fn=spread)
    fb = agg(outputs="abc")
    assert fb.value == 0  # both return 3
    assert fb.name == "spread"


def test_ensemble_feedbacks_mode():
    def count_feedbacks(feedbacks):
        return len(feedbacks)

    agg = make_scorer_ensemble(
        name="count", scorers=[_always_true, _always_false], ensemble_fn=count_feedbacks
    )
    assert agg(outputs="x").value == 2


def test_ensemble_rejects_unsupported_value_type():
    # A sub-scorer whose Feedback carries a non-str/numeric value (dict) is rejected in
    # values-mode. The value is wrapped in a Feedback so it survives Scorer.run() validation
    # and reaches the ensemble's own value-type guard.
    @scorer
    def _returns_dict(outputs) -> Feedback:
        return Feedback(value={"key": "val"})

    agg = make_scorer_ensemble(name="bad", scorers=[_returns_dict], ensemble_fn="majority_vote")
    with pytest.raises(MlflowException, match="_returns_dict"):
        agg(outputs="x")


def test_ensemble_rejects_empty_scorers():
    with pytest.raises(MlflowException, match="at least one"):
        make_scorer_ensemble(name="e", scorers=[], ensemble_fn="mean")


def test_ensemble_rejects_unknown_builtin_name():
    with pytest.raises(MlflowException, match="not a built-in"):
        make_scorer_ensemble(name="e", scorers=[_always_true], ensemble_fn="bogus")


def test_ensemble_session_level_derivation_and_mixing():
    @scorer
    def _sess(session) -> bool:
        return len(session) > 0

    session_agg = make_scorer_ensemble(name="s", scorers=[_sess], ensemble_fn="agg_all")
    assert session_agg.is_session_level_scorer is True

    single_agg = make_scorer_ensemble(name="t", scorers=[_always_true], ensemble_fn="agg_all")
    assert single_agg.is_session_level_scorer is False

    with pytest.raises(MlflowException, match="same level"):
        make_scorer_ensemble(name="m", scorers=[_sess, _always_true], ensemble_fn="agg_all")


def test_ensemble_records_builtin_name_for_callable():
    agg = make_scorer_ensemble(name="v", scorers=[_always_true], ensemble_fn=majority_vote)
    assert agg._ensemble_fn_name == "majority_vote"


def test_ensemble_custom_callable_has_no_builtin_name():
    agg = make_scorer_ensemble(
        name="c", scorers=[_always_true], ensemble_fn=lambda values: max(values)
    )
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
    agg = make_scorer_ensemble(
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
    agg = make_scorer_ensemble(
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
    agg = make_scorer_ensemble(
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
    agg = make_scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: sum(values))
    with pytest.raises(MlflowException, match="custom ensemble function"):
        agg.model_dump()


def test_public_exports():
    assert mlflow.genai.make_scorer_ensemble is make_scorer_ensemble
    for name, fn in [
        ("majority_vote", majority_vote),
        ("mean", mean),
        ("minimum", minimum),
        ("maximum", maximum),
        ("agg_all", agg_all),
        ("agg_any", agg_any),
        ("make_scorer_ensemble", make_scorer_ensemble),
    ]:
        assert getattr(_public_scorers, name) is fn


def test_ensemble_check_can_be_registered_passes_for_builtin_sub_scorers():
    # An ensemble of builtin sub-scorers is registerable: _check_can_be_registered must not raise.
    agg = make_scorer_ensemble(name="ok", scorers=[Safety()], ensemble_fn="majority_vote")
    agg._check_can_be_registered()


def test_ensemble_check_can_be_registered_rejects_decorator_sub_scorer_non_databricks():
    # A decorator sub-scorer is only registerable under a Databricks tracking URI. The ensemble
    # must be rejected with the same rule (recursion into sub-scorers), not silently allowed.
    agg = make_scorer_ensemble(name="e", scorers=[_always_true], ensemble_fn="agg_all")
    with pytest.raises(MlflowException, match="Custom scorer registration"):
        agg._check_can_be_registered()


def test_ensemble_in_evaluate(is_in_databricks):
    data = pd.DataFrame({
        "inputs": [{"q": "a"}, {"q": "b"}],
        "outputs": ["yes", "no"],
    })
    agg = make_scorer_ensemble(
        name="ensemble",
        scorers=[_always_true, _always_false],
        ensemble_fn="agg_any",
    )
    result = mlflow.genai.evaluate(data=data, scorers=[agg])
    # agg_any is order-independent, so the ensemble feedback is logged for every row
    assert "ensemble/mean" in result.metrics


@pytest.mark.parametrize(
    ("scorers", "expected_names"),
    [
        pytest.param([_always_true], ["_always_true"], id="plain"),
        pytest.param(
            [make_scorer_ensemble(name="e", scorers=[_always_true], ensemble_fn="agg_all")],
            ["_always_true"],
            id="ensemble",
        ),
        pytest.param(
            [
                make_scorer_ensemble(
                    name="outer",
                    scorers=[
                        make_scorer_ensemble(
                            name="inner", scorers=[_always_true], ensemble_fn="agg_all"
                        ),
                        _always_false,
                    ],
                    ensemble_fn="agg_all",
                )
            ],
            ["_always_true", "_always_false"],
            id="nested-ensemble",
        ),
    ],
)
def test_flatten_scorers_expands_ensembles(scorers, expected_names):
    assert [s.name for s in flatten_scorers(scorers)] == expected_names


def test_ensemble_sub_scorer_missing_columns_is_reported_upfront(is_in_databricks):
    # Correctness requires `expectations`, which this dataset lacks. The warning must fire
    # for a sub-scorer wrapped in an ensemble, not just a top-level built-in scorer.
    data = pd.DataFrame({"inputs": [{"q": "a"}], "outputs": ["a"]})
    agg = make_scorer_ensemble(
        name="ensemble", scorers=[Correctness()], ensemble_fn="majority_vote"
    )
    with mock.patch("mlflow.genai.evaluation.base.valid_data_for_builtin_scorers") as mock_valid:
        try:
            mlflow.genai.evaluate(data=data, scorers=[agg])
        except Exception:
            pass
    mock_valid.assert_called_once()
    passed_scorers = mock_valid.call_args[0][1]
    assert [type(s).__name__ for s in passed_scorers] == ["Correctness"]


# ---------------------------------------------------------------------------
# Categorical / Literal values and feedback_value_type validation
# ---------------------------------------------------------------------------


@scorer
def _label(outputs) -> str:
    return outputs


def test_ensemble_majority_vote_categorical():
    # majority_vote now handles categorical string values.
    agg = make_scorer_ensemble(
        name="cat", scorers=[_label, _label, _label], ensemble_fn="majority_vote"
    )
    assert agg(outputs="yes").value == "yes"


def test_mean_rejects_categorical_values():
    # mean must raise for string values at call time, not silently crash in statistics.mean.
    agg = make_scorer_ensemble(name="m", scorers=[_label], ensemble_fn="mean")
    with pytest.raises(MlflowException, match="numeric"):
        agg(outputs="0.5")


def test_feedbacks_mode_allows_non_numeric():
    # feedbacks-mode bypasses the value-type check entirely; feedbacks contains Feedback objects.
    def first_value(feedbacks):
        return feedbacks[0].value

    agg = make_scorer_ensemble(name="f", scorers=[_label], ensemble_fn=first_value)
    assert agg(outputs="hello").value == "hello"


def test_ensemble_feedbacks_mode_receives_feedback_objects():
    captured = {}

    def inspect_fn(feedbacks):
        captured["types"] = [type(f).__name__ for f in feedbacks]
        captured["names"] = [f.name for f in feedbacks]
        return feedbacks[0].value

    # _num_len returns a bare int; must be wrapped into a Feedback in feedbacks-mode.
    agg = make_scorer_ensemble(name="fb", scorers=[_num_len], ensemble_fn=inspect_fn)
    fb = agg(outputs="abcd")
    assert captured["types"] == ["Feedback"]
    assert captured["names"] == ["_num_len"]
    assert fb.value == 4


def test_numeric_builtin_rejects_literal_judge_upfront():
    # Safety declares feedback_value_type == Literal["yes","no"]; mean must reject it early.
    with pytest.raises(MlflowException, match="numeric"):
        make_scorer_ensemble(name="x", scorers=[Safety()], ensemble_fn="mean")


def test_majority_vote_accepts_literal_judge():
    # majority_vote is categorical-friendly; Safety() constructs fine.
    agg = make_scorer_ensemble(name="ok", scorers=[Safety()], ensemble_fn="majority_vote")
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


@pytest.mark.parametrize(
    ("feedback_type", "expected"),
    [
        pytest.param(bool, True, id="bool"),
        pytest.param(Literal[True, False], True, id="literal_bool"),
        pytest.param(int, False, id="int"),
        pytest.param(float, False, id="float"),
        pytest.param(Literal[1, 2, 3], False, id="literal_numeric"),
        # Built-in judges declare Literal["yes","no"], which has a yes/no reading and so
        # is usable with the boolean reducers.
        pytest.param(Literal["yes", "no"], True, id="literal_yes_no"),
        pytest.param(Literal["pass", "fail"], True, id="literal_pass_fail"),
        pytest.param(Literal["great", "terrible"], False, id="literal_unmappable_str"),
        pytest.param(str, False, id="str"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_bool_feedback_type(feedback_type, expected):
    assert is_bool_feedback_type(feedback_type) is expected


def test_bool_builtin_accepts_literal_yes_no_judge():
    # "all judges must pass" over built-in judges is the primary agg_all use case. Safety
    # declares Literal["yes","no"], which has a yes/no reading, so it must not be rejected.
    agg = make_scorer_ensemble(name="x", scorers=[Safety()], ensemble_fn="agg_all")
    assert agg.name == "x"


def test_bool_builtin_rejects_non_boolean_literal_judge_upfront():
    judge = make_judge(
        name="grade",
        instructions="Grade {{ outputs }}.",
        model="openai:/gpt-4",
        feedback_value_type=Literal["great", "terrible"],
    )
    with pytest.raises(MlflowException, match="yes/no reading"):
        make_scorer_ensemble(name="x", scorers=[judge], ensemble_fn="agg_all")


def test_agg_all_coerces_yes_no_at_call_time():
    # A decorator scorer declares no feedback_value_type, so the up-front check is skipped.
    # "yes" must still read as True rather than relying on string truthiness.
    agg = make_scorer_ensemble(name="a", scorers=[_label], ensemble_fn="agg_all")
    assert agg(outputs="yes").value is True
    assert agg(outputs="no").value is False


def test_agg_all_at_call_time_rejects_values_without_yes_no_reading():
    agg = make_scorer_ensemble(name="a", scorers=[_label], ensemble_fn="agg_all")
    with pytest.raises(MlflowException, match="yes/no reading"):
        agg(outputs="maybe")


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

    agg = make_scorer_ensemble(name="e", scorers=[_yes, _no], ensemble_fn="majority_vote")
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

    agg = make_scorer_ensemble(name="e", scorers=[_judge], ensemble_fn="agg_all")
    fb = agg(outputs="x")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["source"]["source_type"] == "LLM_JUDGE"
    assert sub[0]["source"]["source_id"] == "gpt-4o-mini"


def test_ensemble_metadata_normalizes_bare_value():
    agg = make_scorer_ensemble(name="c", scorers=[_num_len], ensemble_fn=lambda values: max(values))
    fb = agg(outputs="abcd")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["assessment_name"] == "_num_len"
    assert sub[0]["feedback"]["value"] == 4
    assert sub[0]["source"]["source_type"] == "CODE"


def test_ensemble_fn_metadata_wins_on_collision():
    key = EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY

    def fn_with_meta(feedbacks):
        return Feedback(value=True, metadata={key: "override"})

    agg = make_scorer_ensemble(name="m", scorers=[_always_true], ensemble_fn=fn_with_meta)
    fb = agg(outputs="x")
    # The ensemble_fn's own metadata wins on key collision.
    assert fb.metadata[key] == "override"


def test_ensemble_metadata_is_string_valued():
    agg = make_scorer_ensemble(name="s", scorers=[_always_true], ensemble_fn="agg_all")
    fb = agg(outputs="x")
    for v in fb.metadata.values():
        assert isinstance(v, str)


def test_ensemble_metadata_captures_error_feedback():
    @scorer
    def _errs(outputs) -> Feedback:
        return Feedback(error=ValueError("boom"))

    def tolerant(feedbacks):
        return True

    agg = make_scorer_ensemble(name="e", scorers=[_errs], ensemble_fn=tolerant)
    fb = agg(outputs="x")
    sub = json.loads(fb.metadata[EnsembleScorer._SUB_FEEDBACKS_METADATA_KEY])
    assert sub[0]["assessment_name"] == "_errs"
    # to_dictionary() puts error info under the nested "feedback.error" key
    assert sub[0]["feedback"]["error"]["error_code"] == "ValueError"
    assert sub[0]["feedback"]["error"]["error_message"] == "boom"
