import json
import logging

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.quality_criteria import (
    build_quality_criteria_tag,
    get_extra_aggregations,
    parse_quality_criteria,
    warn_on_unmeasured_quality_criteria,
)
from mlflow.genai.scorers import Correctness, get_scorer, make_scorer_ensemble, scorer
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH


def _make_scorer(name):
    @scorer(name=name)
    def _scorer(outputs) -> float:
        return 1.0

    return _scorer


def _parse(scorers, quality_criteria):
    return [rule.to_dict() for rule in parse_quality_criteria(scorers, quality_criteria)]


@pytest.mark.parametrize("quality_criteria", [None, "", "  "])
def test_parse_returns_no_rules_without_criteria(quality_criteria):
    assert parse_quality_criteria([_make_scorer("a")], quality_criteria) == []


@pytest.mark.parametrize(
    ("quality_criteria", "expected"),
    [
        ("a >= 0.9", [("a/mean", "GTE", 0.9)]),
        ("a <= 2", [("a/mean", "LTE", 2.0)]),
        ("`a/p90` <= -1e-2", [("a/p90", "LTE", -0.01)]),
        ('"a/min" >= 0.5', [("a/min", "GTE", 0.5)]),
        ("a >= 0.5 and `a/max` <= 0.9", [("a/mean", "GTE", 0.5), ("a/max", "LTE", 0.9)]),
        ("`my scorer` >= 0.5", [("my scorer/mean", "GTE", 0.5)]),
        ("`team/latency/max` <= 2", [("team/latency/max", "LTE", 2.0)]),
        ("length <= 10", [("length/mean", "LTE", 10.0)]),
    ],
)
def test_parse_builds_rules(quality_criteria, expected):
    scorers = [
        _make_scorer("a"),
        _make_scorer("my scorer"),
        _make_scorer("team/latency"),
        _make_scorer("length"),
    ]
    rules = _parse(scorers, quality_criteria)
    assert [(r["metricKey"], r["comparator"], r["threshold"]) for r in rules] == expected


def test_parse_prefers_scorer_name_containing_aggregation():
    rules = _parse([_make_scorer("a"), _make_scorer("a/min")], "`a/min` >= 0.5")
    assert rules == [{"metricKey": "a/min/mean", "comparator": "GTE", "threshold": 0.5}]


@pytest.mark.parametrize(
    "quality_criteria",
    [
        "a >= 0.5 OR a <= 0.9",
        "a >= 0.5 AND",
        "AND a >= 0.5",
        "a >= 0.5 a <= 0.9",
        "(a >= 0.5)",
        "NOT a >= 0.5",
        "a >= 0.5; a <= 0.9",
        "a >= '0.5'",
        "0.5 <= a",
        "a/min >= 0.5",
        "a >= 0.5AND a <= 0.9",
        "a >= 0.5x",
        "a",
    ],
)
def test_parse_rejects_invalid_syntax(quality_criteria):
    with pytest.raises(MlflowException, match="Invalid quality criteria"):
        parse_quality_criteria([_make_scorer("a")], quality_criteria)


@pytest.mark.parametrize("comparator", [">", "<", "=", "!="])
def test_parse_rejects_unsupported_comparators(comparator):
    with pytest.raises(MlflowException, match="only `>=` and `<=` are supported"):
        parse_quality_criteria([_make_scorer("a")], f"a {comparator} 0.5")


def test_parse_rejects_non_finite_threshold():
    with pytest.raises(MlflowException, match="must be a finite number"):
        parse_quality_criteria([_make_scorer("a")], "a >= 1e999")


def test_parse_rejects_non_string():
    with pytest.raises(MlflowException, match="must be a string such as"):
        parse_quality_criteria([_make_scorer("a")], {"a": 0.5})


@pytest.mark.parametrize(
    ("quality_criteria", "key"),
    [("unknown >= 0.5", "unknown"), ("`correctness/sum` >= 0.5", "correctness/sum")],
)
def test_parse_rejects_condition_matching_no_scorer(quality_criteria, key):
    scorers = [_make_scorer("correctness"), _make_scorer("safety")]
    with pytest.raises(
        MlflowException,
        match=rf"'{key}' doesn't match any scorer .* Scorer names: 'correctness', 'safety'",
    ):
        parse_quality_criteria(scorers, quality_criteria)


def test_parse_rejects_condition_matching_duplicate_scorer_names():
    scorers = [_make_scorer("a"), _make_scorer("a")]
    with pytest.raises(MlflowException, match="'a' matches more than one scorer"):
        parse_quality_criteria(scorers, "a >= 0.5")


def test_parse_allows_duplicate_names_without_criteria():
    scorers = [_make_scorer("a"), _make_scorer("a"), _make_scorer("b")]
    [rule] = _parse(scorers, "b >= 0.5")
    assert rule["metricKey"] == "b/mean"


def test_parse_accepts_ensemble_and_rejects_its_sub_scorer():
    ensemble = make_scorer_ensemble(
        name="ensemble",
        scorers=[_make_scorer("sub"), _make_scorer("other")],
        ensemble_fn="mean",
    )

    [rule] = _parse([ensemble], "ensemble >= 0.7")
    assert rule["metricKey"] == "ensemble/mean"
    with pytest.raises(MlflowException, match="'sub' doesn't match any scorer"):
        parse_quality_criteria([ensemble], "sub >= 0.7")


def test_parse_matches_registered_scorer_by_registered_name():
    experiment_id = mlflow.create_experiment("test_quality_criteria_registered_name")
    Correctness().register(name="registered_correctness", experiment_id=experiment_id)
    loaded = get_scorer(name="registered_correctness", experiment_id=experiment_id)

    rules = _parse([loaded], "registered_correctness >= 0.8")

    assert rules == [
        {"metricKey": "registered_correctness/mean", "comparator": "GTE", "threshold": 0.8}
    ]


def test_parse_skips_scorers_without_kind():
    # Legacy Databricks `Metric`s are accepted by `validate_scorers` but have no `kind`.
    class LegacyMetric:
        name = "legacy"

    [rule] = _parse([LegacyMetric(), _make_scorer("a")], "a >= 0.5")
    assert rule["metricKey"] == "a/mean"


def test_get_extra_aggregations():
    rules = parse_quality_criteria(
        [_make_scorer("a"), _make_scorer("b")], "a >= 0.5 AND `a/p90` <= 0.9 AND `b/min` >= 0.1"
    )
    assert get_extra_aggregations(rules) == {"a": ["mean", "p90"], "b": ["min"]}


def test_build_tag_returns_none_without_rules():
    assert build_quality_criteria_tag([]) is None


def test_build_tag_serializes_rules_in_ui_criteria_shape():
    scorers = [_make_scorer("correctness"), _make_scorer("latency"), _make_scorer("no_threshold")]
    rules = parse_quality_criteria(scorers, "correctness >= 0.9 AND `latency/p90` <= 2")

    payload = json.loads(build_quality_criteria_tag(rules))

    assert payload == {
        "schemaVersion": 1,
        "revision": payload["revision"],
        "rules": [
            {"metricKey": "correctness/mean", "comparator": "GTE", "threshold": 0.9},
            {"metricKey": "latency/p90", "comparator": "LTE", "threshold": 2.0},
        ],
    }


def test_build_tag_uses_fresh_revision_per_call():
    rules = parse_quality_criteria([_make_scorer("a")], "a >= 0.5")
    first = json.loads(build_quality_criteria_tag(rules))
    second = json.loads(build_quality_criteria_tag(rules))
    assert first["revision"] != second["revision"]


def test_build_tag_rejects_oversized_payload():
    names = ["s" * 200 + str(i) for i in range(40)]
    rules = parse_quality_criteria(
        [_make_scorer(name) for name in names], " AND ".join(f"{name} >= 0.5" for name in names)
    )
    with pytest.raises(MlflowException, match=f"over the {MAX_TAG_VAL_LENGTH} character limit"):
        build_quality_criteria_tag(rules)


def test_warn_on_unmeasured_quality_criteria(caplog):
    scorers = [_make_scorer("measured"), _make_scorer("unmeasured")]
    rules = parse_quality_criteria(scorers, "measured >= 0.5 AND `unmeasured/p90` >= 0.5")

    with caplog.at_level(logging.WARNING, logger="mlflow.genai.evaluation.quality_criteria"):
        warn_on_unmeasured_quality_criteria(rules, {"measured/mean": 0.7, "unmeasured/mean": 0.7})

    [record] = caplog.records
    assert "scorer 'unmeasured'" in record.message
    assert "'unmeasured/p90'" in record.message
