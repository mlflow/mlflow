from typing import Any
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.entities.dataset_record import DATASET_RECORD_SCORERS_TAG
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _resolve_per_record_scorers, _scorers_for_item
from mlflow.genai.scorers.base import Scorer, scorer


@scorer
def always_true(outputs) -> bool:
    return True


@scorer
def always_false(outputs) -> bool:
    return False


def _values(result, scorer_name: str) -> list[Any]:
    return result.result_df[f"{scorer_name}/value"].tolist()


def _assessment_names_by_input(run_id: str) -> dict[str, set[str]]:
    """Map each trace's `q` input to the set of assessment names on that trace."""
    traces = mlflow.search_traces(run_id=run_id, return_type="list")
    return {
        trace.data.spans[0].inputs["q"]: {a.name for a in trace.info.assessments}
        for trace in traces
    }


def _patch_resolution(scorers: list[Scorer]) -> mock._patch:
    """Resolve per-record names to the given scorers.

    Registering a @scorer function requires a Databricks tracking URI, so patching the
    resolver is the only way to exercise a deterministic custom scorer per record.
    """
    return mock.patch(
        "mlflow.genai.evaluation.harness.resolve_scorer_names",
        return_value={s.name: s for s in scorers},
    )


def test_record_scorer_runs_only_on_its_own_row():
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "scorers": ["always_false"]},
        {"inputs": {"q": "b"}, "outputs": "y"},
    ]

    with _patch_resolution([always_false]):
        result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    assert _assessment_names_by_input(result.run_id) == {
        "a": {"always_true", "always_false"},
        "b": {"always_true"},
    }
    # The row the scorer did not run on has no value, rather than a default one.
    values = _values(result, "always_false")
    assert values[0] is False
    assert pd.isna(values[1])


def test_record_scorer_runs_with_a_real_builtin_name():
    # End-to-end resolution against the built-in scorers, with no patching.
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "scorers": ["safety"]},
        {"inputs": {"q": "b"}, "outputs": "y"},
    ]

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    # `safety` needs an LLM, so it may error here; what matters is that it was invoked on
    # row "a" alone.
    assert _assessment_names_by_input(result.run_id) == {
        "a": {"always_true", "safety"},
        "b": {"always_true"},
    }


def test_run_level_scorers_still_apply_to_every_row():
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "scorers": ["safety"]},
        {"inputs": {"q": "b"}, "outputs": "y"},
    ]

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    assert _values(result, "always_true") == [True, True]


def test_record_without_scorers_is_unaffected():
    data = [{"inputs": {"q": "a"}, "outputs": "x"}]

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    assert _values(result, "always_true") == [True]
    assert "safety/value" not in result.result_df.columns


def test_evaluate_with_only_record_scorers_and_no_run_level_list():
    data = [{"inputs": {"q": "a"}, "outputs": "x", "scorers": ["always_true"]}]

    with _patch_resolution([always_true]):
        result = mlflow.genai.evaluate(data=data)

    assert _values(result, "always_true") == [True]


def test_duplicate_between_record_and_run_level_scorers_runs_once():
    data = [{"inputs": {"q": "a"}, "outputs": "x", "scorers": ["always_true"]}]

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    feedbacks = [a for a in traces[0].info.assessments if a.name == "always_true"]
    assert len(feedbacks) == 1


def test_unresolvable_record_scorer_name_raises():
    data = [{"inputs": {"q": "a"}, "outputs": "x", "scorers": ["does_not_exist"]}]

    with pytest.raises(MlflowException, match=r"could not be resolved: \['does_not_exist'\]"):
        mlflow.genai.evaluate(data=data, scorers=[always_true])


def test_record_scorer_object_is_rejected():
    data = [{"inputs": {"q": "a"}, "outputs": "x", "scorers": [always_true]}]

    with pytest.raises(MlflowException, match=r"Pass the scorer's registered name"):
        mlflow.genai.evaluate(data=data)


def test_record_scorers_accepted_as_dataframe_column():
    data = pd.DataFrame([
        {"inputs": {"q": "a"}, "outputs": "x", "scorers": ["safety"]},
        {"inputs": {"q": "b"}, "outputs": "y", "scorers": None},
    ])

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    assert _values(result, "always_true") == [True, True]


def test_reserved_scorers_tag_is_not_copied_to_the_trace():
    data = [
        {
            "inputs": {"q": "a"},
            "outputs": "x",
            "tags": {"team": "search"},
            "scorers": ["safety"],
        }
    ]

    result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert traces[0].info.tags["team"] == "search"
    assert DATASET_RECORD_SCORERS_TAG not in traces[0].info.tags


def test_per_record_scorer_metrics_are_aggregated_over_its_own_rows():
    # A per-record scorer's mean uses the rows it ran on, not the full row count.
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "scorers": ["always_false"]},
        {"inputs": {"q": "b"}, "outputs": "y"},
        {"inputs": {"q": "c"}, "outputs": "z"},
    ]

    with _patch_resolution([always_false]):
        result = mlflow.genai.evaluate(data=data, scorers=[always_true])

    # always_false ran on one row and returned False, so its mean is 0.0 rather than the
    # 2/3 that averaging over all three rows would give.
    assert result.metrics["always_false/mean"] == 0.0
    assert result.metrics["always_true/mean"] == 1.0


def test_persisted_dataset_round_trips_record_scorers():
    dataset = create_dataset(name="per_record_scorers_ds")
    dataset.merge_records([
        {"inputs": {"q": "a"}, "scorers": ["safety", "correctness"]},
        {"inputs": {"q": "b"}},
    ])

    records = {r["inputs"]["q"]: r for r in dataset.to_df().to_dict("records")}
    assert records["a"]["tags"][DATASET_RECORD_SCORERS_TAG] == "safety,correctness"
    assert DATASET_RECORD_SCORERS_TAG not in records["b"]["tags"]

    assert EvalItem.from_dataset_row(records["a"]).scorers == ["safety", "correctness"]
    assert EvalItem.from_dataset_row(records["b"]).scorers == []


def test_merge_records_rejects_a_scorer_object():
    dataset = create_dataset(name="per_record_scorers_reject_ds")

    with pytest.raises(MlflowException, match=r"Pass the scorer's registered name"):
        dataset.merge_records([{"inputs": {"q": "a"}, "scorers": [always_true]}])


def test_eval_item_scorers_column_overrides_the_persisted_tag():
    row = {
        "inputs": {"q": "a"},
        "tags": {DATASET_RECORD_SCORERS_TAG: "safety"},
        "scorers": ["correctness"],
    }

    assert EvalItem.from_dataset_row(row).scorers == ["correctness"]


def _eval_item(scorers=None) -> EvalItem:
    return EvalItem(
        request_id="req-1",
        inputs={"q": "a"},
        outputs="x",
        expectations={},
        scorers=scorers or [],
    )


def test_scorers_for_item_returns_run_level_list_when_record_has_none():
    run_level = [always_true]

    assert _scorers_for_item(_eval_item(), run_level, {"safety": always_false}) is run_level


def test_scorers_for_item_appends_record_scorers():
    resolved = _scorers_for_item(
        _eval_item(["always_false"]), [always_true], {"always_false": always_false}
    )

    assert [s.name for s in resolved] == ["always_true", "always_false"]


def test_scorers_for_item_deduplicates_by_name():
    resolved = _scorers_for_item(
        _eval_item(["always_true"]), [always_true], {"always_true": always_true}
    )

    assert [s.name for s in resolved] == ["always_true"]


def test_scorers_for_item_skips_names_that_did_not_resolve():
    resolved = _scorers_for_item(_eval_item(["missing"]), [always_true], {})

    assert [s.name for s in resolved] == ["always_true"]


def test_resolve_per_record_scorers_skips_run_level_names():
    resolved = _resolve_per_record_scorers(
        [_eval_item(["always_true"])], [always_true], experiment_id=None
    )

    # always_true is already a run-level scorer, so no registry lookup is needed for it.
    assert resolved == {}


def test_resolve_per_record_scorers_resolves_builtins():
    resolved = _resolve_per_record_scorers([_eval_item(["safety"])], [], experiment_id=None)

    assert set(resolved) == {"safety"}
