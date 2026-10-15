"""Quality thresholds passed to ``evaluate``, recorded on the evaluation run as a tag."""

import json
import logging
import uuid
from collections import Counter
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import QualityThreshold, Scorer
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

# NB: The payload follows the criteria JSON the evaluation runs UI already reads.
_SCHEMA_VERSION = 1
_COMPARATORS = {"at_least": "GTE", "at_most": "LTE"}


def build_quality_threshold_rules(
    scorers: list[Scorer], quality_thresholds: dict[str, QualityThreshold] | None
) -> list[dict[str, Any]]:
    """
    Validate ``quality_thresholds`` against ``scorers`` and build one rule per threshold.

    Each key must be the name of exactly one scorer in ``scorers``.
    """
    if not quality_thresholds:
        return []
    if not isinstance(quality_thresholds, dict):
        raise MlflowException.invalid_parameter_value(
            "`quality_thresholds` must be a dict that maps a scorer name to a "
            f"`QualityThreshold`, got {type(quality_thresholds).__name__}."
        )

    # NB: `getattr`, because legacy Databricks `Metric` objects aren't scorers.
    name_counts = Counter(getattr(scorer, "name", None) for scorer in scorers)
    rules = []
    for name, threshold in quality_thresholds.items():
        if not isinstance(threshold, QualityThreshold):
            raise MlflowException.invalid_parameter_value(
                f"The quality threshold for {name!r} must be a `QualityThreshold`, "
                f"got {type(threshold).__name__}."
            )
        if name_counts[name] == 0:
            scorer_names = ", ".join(repr(n) for n in sorted(n for n in name_counts if n))
            raise MlflowException.invalid_parameter_value(
                f"The quality threshold for {name!r} doesn't match any scorer passed to "
                f"`evaluate`. Scorer names: {scorer_names}."
            )
        if name_counts[name] > 1:
            raise MlflowException.invalid_parameter_value(
                f"The quality threshold for {name!r} matches more than one scorer passed to "
                "`evaluate`, so it can't be matched to a single metric. Give each scorer a "
                "unique name."
            )
        bound = "at_least" if threshold.at_least is not None else "at_most"
        rules.append({
            "metricKey": f"{name}/{threshold.aggregation}",
            "comparator": _COMPARATORS[bound],
            "threshold": getattr(threshold, bound),
        })
    return rules


def build_quality_thresholds_tag(rules: list[dict[str, Any]]) -> str | None:
    """
    Serialize the rules from ``build_quality_threshold_rules`` for the run tag.

    Returns:
        The tag value, or None when there are no rules.
    """
    if not rules:
        return None

    value = json.dumps(
        {
            "schemaVersion": _SCHEMA_VERSION,
            "revision": str(uuid.uuid4()),
            "rules": rules,
        },
        separators=(",", ":"),
    )
    if len(value) > MAX_TAG_VAL_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"The quality thresholds are {len(value)} characters once serialized, over the "
            f"{MAX_TAG_VAL_LENGTH} character limit for a run tag. Set fewer quality "
            "thresholds."
        )
    return value


def warn_on_unmeasured_quality_thresholds(
    rules: list[dict[str, Any]], aggregated_metrics: dict[str, float]
) -> None:
    """
    Warn for each threshold whose metric this evaluation did not produce.

    This happens when a scorer returns non-numeric values, emits Feedback under a different
    name than its own, or fails on every row.
    """
    for rule in rules:
        metric_key = rule["metricKey"]
        if metric_key not in aggregated_metrics:
            scorer_name = metric_key.rsplit("/", 1)[0]
            _logger.warning(
                f"The quality threshold on scorer '{scorer_name}' can't be checked: this "
                f"evaluation produced no '{metric_key}' metric. The scorer must return booleans, "
                "numbers, or yes/no values under its own name. If the run already has this "
                "metric from an earlier evaluation, the threshold is compared against that value."
            )
