from typing import Any

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _scorers_for_item
from mlflow.genai.evaluation.utils import row_matches_scorer_filter
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import Safety, UserFrustration


@scorer
def always_true(outputs) -> bool:
    return True


@scorer
def always_false(outputs) -> bool:
    return False


def _values(result, scorer_name: str) -> list[Any]:
    return result.result_df[f"{scorer_name}/value"].tolist()


# ---- row_matches_scorer_filter ------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "filter_string", "expected"),
    [
        ({"category": "billing"}, "tags.`category` = 'billing'", True),
        ({"category": "greeting"}, "tags.`category` = 'billing'", False),
        ({}, "tags.`category` = 'billing'", False),
        ({"category": "greeting"}, "tags.`category` != 'billing'", True),
        ({"pii": "x"}, "tags.`pii` IS NOT NULL", True),
        ({"other": "x"}, "tags.`pii` IS NULL", True),
        (
            {"category": "billing", "reviewed": "true"},
            "tags.`category` = 'billing' AND tags.`reviewed` = 'true'",
            True,
        ),
        (
            {"category": "billing"},
            "tags.`category` = 'billing' AND tags.`reviewed` = 'true'",
            False,
        ),
    ],
)
def test_row_matches_scorer_filter(tags: dict[str, str], filter_string: str, expected: bool):
    assert row_matches_scorer_filter(tags, filter_string) is expected


def test_row_matches_scorer_filter_none_tags():
    assert row_matches_scorer_filter(None, "tags.`category` = 'billing'") is False


def test_row_matches_scorer_filter_rejects_non_tag_clause():
    with pytest.raises(MlflowException, match="may only reference tags"):
        row_matches_scorer_filter({"category": "billing"}, "trace.status = 'OK'")


# ---- Scorer.where -------------------------------------------------------------


def test_where_returns_copy_and_sets_filter():
    scoped = always_true.where("tags.`category` = 'billing'")

    assert scoped is not always_true
    assert always_true.row_filter is None
    assert scoped.row_filter == "tags.`category` = 'billing'"
    assert scoped.name == always_true.name


def test_where_on_builtin_scorer():
    scoped = Safety().where("tags.`pii` = 'true'")

    assert isinstance(scoped, Safety)
    assert scoped.row_filter == "tags.`pii` = 'true'"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_where_rejects_empty_filter(bad):
    with pytest.raises(MlflowException, match="must be a non-empty string"):
        always_true.where(bad)


def test_where_rejects_session_level_scorer():
    with pytest.raises(MlflowException, match="session-level scorer"):
        UserFrustration().where("tags.`x` = 'y'")


# ---- _scorers_for_item --------------------------------------------------------


def _eval_item(tags=None) -> EvalItem:
    return EvalItem(request_id="r", inputs={"q": "a"}, outputs="x", expectations={}, tags=tags)


def test_scorers_for_item_returns_same_list_when_no_filters():
    scorers = [always_true, always_false]

    assert _scorers_for_item(_eval_item({"category": "billing"}), scorers) is scorers


def test_scorers_for_item_keeps_matching_scoped_scorer():
    scoped = always_false.where("tags.`category` = 'billing'")

    result = _scorers_for_item(_eval_item({"category": "billing"}), [always_true, scoped])

    assert [s.name for s in result] == ["always_true", "always_false"]


def test_scorers_for_item_drops_non_matching_scoped_scorer():
    scoped = always_false.where("tags.`category` = 'billing'")

    result = _scorers_for_item(_eval_item({"category": "greeting"}), [always_true, scoped])

    assert [s.name for s in result] == ["always_true"]


def test_scorers_for_item_untagged_row_only_runs_unscoped():
    scoped = always_false.where("tags.`category` = 'billing'")

    result = _scorers_for_item(_eval_item(tags=None), [always_true, scoped])

    assert [s.name for s in result] == ["always_true"]


# ---- end-to-end ---------------------------------------------------------------


def test_scoped_scorer_runs_only_on_matching_rows():
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "tags": {"category": "billing"}},
        {"inputs": {"q": "b"}, "outputs": "y", "tags": {"category": "greeting"}},
    ]

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[always_true, always_false.where("tags.`category` = 'billing'")],
    )

    # The unscoped scorer ran on every row; the scoped one on exactly the one matching row.
    assert _values(result, "always_true") == [True, True]
    scoped_values = _values(result, "always_false")
    assert [v for v in scoped_values if not pd.isna(v)] == [False]


def test_scoped_scorer_metric_aggregates_over_matched_rows_only():
    data = [
        {"inputs": {"q": "a"}, "outputs": "x", "tags": {"category": "billing"}},
        {"inputs": {"q": "b"}, "outputs": "y", "tags": {"category": "greeting"}},
        {"inputs": {"q": "c"}, "outputs": "z", "tags": {"category": "greeting"}},
    ]

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[always_true, always_false.where("tags.`category` = 'billing'")],
    )

    # always_false ran on the one billing row and returned False, so its mean is 0.0 —
    # not diluted by the two rows it never ran on.
    assert result.metrics["always_false/mean"] == 0.0
    assert result.metrics["always_true/mean"] == 1.0


def test_scoped_scorer_matches_dataframe_tags_column():
    data = pd.DataFrame([
        {"inputs": {"q": "a"}, "outputs": "x", "tags": {"category": "billing"}},
        {"inputs": {"q": "b"}, "outputs": "y", "tags": {"category": "greeting"}},
    ])

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[always_true, always_false.where("tags.`category` = 'billing'")],
    )

    assert result.metrics["always_false/mean"] == 0.0


def test_unresolvable_filter_clause_errors_at_evaluate():
    data = [{"inputs": {"q": "a"}, "outputs": "x", "tags": {"category": "billing"}}]

    with pytest.raises(MlflowException, match="may only reference tags"):
        mlflow.genai.evaluate(
            data=data,
            scorers=[always_false.where("trace.status = 'OK'")],
        )


def test_scoped_scorer_matches_trace_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("tag-scoped-trace")

    @mlflow.trace
    def app(q):
        mlflow.update_current_trace(tags={"category": "billing" if q == "a" else "greeting"})
        return "refund" if q == "a" else "hi"

    trace_ids = []
    for q in ("a", "b"):
        app(q)
        trace_ids.append(mlflow.get_last_active_trace_id())
    traces = [mlflow.get_trace(tid, flush=True) for tid in trace_ids]

    # A list of Trace objects flows through `traces_to_df`, which surfaces each trace's tags
    # as the row's `tags`, so `.where()` matches on them.
    result = mlflow.genai.evaluate(
        data=traces,
        scorers=[always_true, always_false.where("tags.`category` = 'billing'")],
    )

    scoped_values = _values(result, "always_false")
    assert [v for v in scoped_values if not pd.isna(v)] == [False]
