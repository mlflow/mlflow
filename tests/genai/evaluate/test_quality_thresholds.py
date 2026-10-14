import json
import logging

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.quality_thresholds import (
    build_quality_threshold_rules,
    build_quality_thresholds_tag,
    warn_on_unmeasured_quality_thresholds,
)
from mlflow.genai.scorers import (
    Correctness,
    QualityThreshold,
    get_scorer,
    make_scorer_ensemble,
    scorer,
)
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH


@pytest.mark.parametrize(
    "kwargs",
    [{"at_least": 0.9}, {"at_most": 2}, {"at_least": 0.5, "aggregation": "p90"}],
)
def test_quality_threshold_accepts_valid_bounds(kwargs):
    assert QualityThreshold(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of `at_least` or `at_most`"),
        ({"at_least": 0.1, "at_most": 0.9}, "exactly one of `at_least` or `at_most`"),
        ({"at_least": True}, "must be a finite number"),
        ({"at_least": "0.9"}, "must be a finite number"),
        ({"at_most": float("nan")}, "must be a finite number"),
        ({"at_least": float("inf")}, "must be a finite number"),
        ({"at_least": 0.9, "aggregation": "sum"}, "`aggregation` must be one of"),
    ],
)
def test_quality_threshold_rejects_invalid_bounds(kwargs, match):
    with pytest.raises(MlflowException, match=match):
        QualityThreshold(**kwargs)


def _make_scorer(name):
    @scorer(name=name)
    def _scorer(outputs) -> float:
        return 1.0

    return _scorer


def _build_tag(scorers, quality_thresholds):
    return build_quality_thresholds_tag(build_quality_threshold_rules(scorers, quality_thresholds))


@pytest.mark.parametrize("quality_thresholds", [None, {}])
def test_build_tag_returns_none_without_thresholds(quality_thresholds):
    assert _build_tag([_make_scorer("a"), _make_scorer("b")], quality_thresholds) is None


def test_build_tag_serializes_rules_in_ui_criteria_shape():
    scorers = [_make_scorer("correctness"), _make_scorer("latency"), _make_scorer("no_threshold")]

    payload = json.loads(
        _build_tag(
            scorers,
            {
                "correctness": QualityThreshold(at_least=0.9),
                "latency": QualityThreshold(at_most=2.0, aggregation="p90"),
            },
        )
    )

    assert payload == {
        "schemaVersion": 1,
        "revision": payload["revision"],
        "rules": [
            {
                "metricKey": "correctness/mean",
                "comparator": "GTE",
                "threshold": 0.9,
            },
            {
                "metricKey": "latency/p90",
                "comparator": "LTE",
                "threshold": 2.0,
            },
        ],
    }


def test_build_tag_uses_fresh_revision_per_call():
    scorers = [_make_scorer("a")]
    thresholds = {"a": QualityThreshold(at_least=0.5)}
    first = json.loads(_build_tag(scorers, thresholds))
    second = json.loads(_build_tag(scorers, thresholds))
    assert first["revision"] != second["revision"]


def test_build_tag_rejects_threshold_matching_no_scorer():
    scorers = [_make_scorer("correctness"), _make_scorer("safety")]
    with pytest.raises(
        MlflowException,
        match=r"'unknown' doesn't match any scorer .* Scorer names: 'correctness', 'safety'",
    ):
        _build_tag(scorers, {"unknown": QualityThreshold(at_least=0.5)})


def test_build_tag_rejects_threshold_matching_duplicate_scorer_names():
    scorers = [_make_scorer("a"), _make_scorer("a")]
    with pytest.raises(MlflowException, match="'a' matches more than one scorer"):
        _build_tag(scorers, {"a": QualityThreshold(at_least=0.5)})


def test_build_tag_allows_duplicate_names_without_thresholds():
    scorers = [_make_scorer("a"), _make_scorer("a"), _make_scorer("b")]
    [rule] = json.loads(_build_tag(scorers, {"b": QualityThreshold(at_least=0.5)}))["rules"]
    assert rule["metricKey"] == "b/mean"


def test_build_tag_accepts_ensemble_and_rejects_its_sub_scorer():
    ensemble = make_scorer_ensemble(
        name="ensemble",
        scorers=[_make_scorer("sub"), _make_scorer("other")],
        ensemble_fn="mean",
    )

    [rule] = json.loads(_build_tag([ensemble], {"ensemble": QualityThreshold(at_least=0.7)}))[
        "rules"
    ]
    assert rule["metricKey"] == "ensemble/mean"
    with pytest.raises(MlflowException, match="'sub' doesn't match any scorer"):
        _build_tag([ensemble], {"sub": QualityThreshold(at_least=0.7)})


def test_build_tag_matches_registered_scorer_by_registered_name():
    experiment_id = mlflow.create_experiment("test_quality_threshold_registered_name")
    Correctness().register(name="registered_correctness", experiment_id=experiment_id)
    loaded = get_scorer(name="registered_correctness", experiment_id=experiment_id)

    rules = build_quality_threshold_rules(
        [loaded], {"registered_correctness": QualityThreshold(at_least=0.8)}
    )

    assert rules == [
        {"metricKey": "registered_correctness/mean", "comparator": "GTE", "threshold": 0.8}
    ]


@pytest.mark.parametrize(
    ("quality_thresholds", "match"),
    [
        ([QualityThreshold(at_least=0.5)], "must be a dict"),
        ({"a": 0.5}, "must be a `QualityThreshold`, got float"),
    ],
)
def test_build_tag_rejects_invalid_types(quality_thresholds, match):
    with pytest.raises(MlflowException, match=match):
        _build_tag([_make_scorer("a")], quality_thresholds)


def test_build_tag_skips_scorers_without_kind():
    # Legacy Databricks `Metric`s are accepted by `validate_scorers` but have no `kind`.
    class LegacyMetric:
        name = "legacy"

    payload = _build_tag([LegacyMetric(), _make_scorer("a")], {"a": QualityThreshold(at_least=0.5)})
    [rule] = json.loads(payload)["rules"]
    assert rule["metricKey"] == "a/mean"


def test_build_tag_rejects_oversized_payload():
    names = ["s" * 200 + str(i) for i in range(40)]
    with pytest.raises(MlflowException, match=f"over the {MAX_TAG_VAL_LENGTH} character limit"):
        _build_tag(
            [_make_scorer(name) for name in names],
            {name: QualityThreshold(at_least=0.5) for name in names},
        )


def test_warn_on_unmeasured_quality_thresholds(caplog):
    scorers = [_make_scorer("measured"), _make_scorer("unmeasured")]
    thresholds = {
        "measured": QualityThreshold(at_least=0.5),
        "unmeasured": QualityThreshold(at_least=0.5, aggregation="p90"),
    }

    with caplog.at_level(logging.WARNING, logger="mlflow.genai.evaluation.quality_thresholds"):
        warn_on_unmeasured_quality_thresholds(
            build_quality_threshold_rules(scorers, thresholds),
            {"measured/mean": 0.7, "unmeasured/mean": 0.7},
        )

    [record] = caplog.records
    assert "scorer 'unmeasured'" in record.message
    assert "'unmeasured/p90'" in record.message
