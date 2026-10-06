import json
import logging

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.quality_thresholds import (
    build_quality_thresholds_tag,
    warn_on_unmeasured_quality_thresholds,
)
from mlflow.genai.scorers import QualityThreshold, make_scorer_ensemble, scorer
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH


def _make_scorer(name, quality_threshold=None):
    @scorer(name=name, quality_threshold=quality_threshold)
    def _scorer(outputs) -> float:
        return 1.0

    return _scorer


def test_build_tag_returns_none_without_thresholds():
    assert build_quality_thresholds_tag([]) is None
    assert build_quality_thresholds_tag([_make_scorer("a"), _make_scorer("b")]) is None


def test_build_tag_serializes_rules_in_ui_criteria_shape():
    scorers = [
        _make_scorer("correctness", 0.9),
        _make_scorer("latency", QualityThreshold(at_most=2.0, aggregation="p90")),
        _make_scorer("no_threshold"),
    ]

    payload = json.loads(build_quality_thresholds_tag(scorers))

    assert payload == {
        "schemaVersion": 1,
        "revision": payload["revision"],
        "aggregation": "ALL",
        "missingMetricBehavior": "INCOMPLETE",
        "rules": [
            {
                "metricKey": "correctness/mean",
                "comparator": "GTE",
                "threshold": 0.9,
                "scorerName": "correctness",
                "scorerAggregation": "mean",
            },
            {
                "metricKey": "latency/p90",
                "comparator": "LTE",
                "threshold": 2.0,
                "scorerName": "latency",
                "scorerAggregation": "p90",
            },
        ],
    }


def test_build_tag_uses_fresh_revision_per_call():
    scorers = [_make_scorer("a", 0.5)]
    first = json.loads(build_quality_thresholds_tag(scorers))
    second = json.loads(build_quality_thresholds_tag(scorers))
    assert first["revision"] != second["revision"]


def test_build_tag_rejects_duplicate_scorer_name_with_threshold():
    scorers = [_make_scorer("a", 0.5), _make_scorer("a")]
    with pytest.raises(MlflowException, match="another scorer passed to `evaluate` has the same"):
        build_quality_thresholds_tag(scorers)


def test_build_tag_allows_duplicate_names_without_thresholds():
    assert build_quality_thresholds_tag([_make_scorer("a"), _make_scorer("a")]) is None


def test_build_tag_rejects_threshold_on_ensemble_sub_scorer():
    ensemble = make_scorer_ensemble(
        name="ensemble",
        scorers=[_make_scorer("sub", 0.5), _make_scorer("other")],
        ensemble_fn="mean",
    )
    with pytest.raises(MlflowException, match="'sub', a sub-scorer of the ensemble 'ensemble'"):
        build_quality_thresholds_tag([ensemble])


def test_build_tag_accepts_threshold_on_ensemble():
    ensemble = make_scorer_ensemble(
        name="ensemble",
        scorers=[_make_scorer("sub"), _make_scorer("other")],
        ensemble_fn="mean",
        quality_threshold=0.7,
    )
    [rule] = json.loads(build_quality_thresholds_tag([ensemble]))["rules"]
    assert rule["metricKey"] == "ensemble/mean"


def test_build_tag_skips_scorers_without_kind():
    # Legacy Databricks `Metric`s are accepted by `validate_scorers` but have no `kind`.
    class LegacyMetric:
        name = "legacy"

    assert build_quality_thresholds_tag([LegacyMetric()]) is None
    payload = build_quality_thresholds_tag([LegacyMetric(), _make_scorer("a", 0.5)])
    [rule] = json.loads(payload)["rules"]
    assert rule["scorerName"] == "a"


def test_build_tag_rejects_oversized_payload():
    scorers = [_make_scorer("s" * 200 + str(i), 0.5) for i in range(40)]
    with pytest.raises(MlflowException, match=f"over the {MAX_TAG_VAL_LENGTH} character limit"):
        build_quality_thresholds_tag(scorers)


def test_warn_on_unmeasured_quality_thresholds(caplog):
    scorers = [
        _make_scorer("measured", 0.5),
        _make_scorer("unmeasured", QualityThreshold(at_least=0.5, aggregation="p90")),
    ]

    with caplog.at_level(logging.WARNING, logger="mlflow.genai.evaluation.quality_thresholds"):
        warn_on_unmeasured_quality_thresholds(
            scorers, {"measured/mean": 0.7, "unmeasured/mean": 0.7}
        )

    [record] = caplog.records
    assert "scorer 'unmeasured'" in record.message
    assert "'unmeasured/p90'" in record.message
