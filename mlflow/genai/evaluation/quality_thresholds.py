"""Quality thresholds declared on scorers, recorded on the evaluation run as a tag."""

import json
import logging
import uuid
from collections import Counter
from typing import Any, Iterator

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer, ScorerKind, _as_quality_threshold
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

# NB: The payload follows the criteria JSON the evaluation runs UI already reads.
_SCHEMA_VERSION = 1
_COMPARATORS = {"at_least": "GTE", "at_most": "LTE"}


def _sub_scorers(scorer: Scorer) -> Iterator[Scorer]:
    # Legacy Databricks `Metric`s are accepted as scorers but have no `kind`.
    if getattr(scorer, "kind", None) == ScorerKind.ENSEMBLE:
        for sub_scorer in scorer._scorers:
            yield sub_scorer
            yield from _sub_scorers(sub_scorer)


def build_quality_threshold_rules(scorers: list[Scorer]) -> list[dict[str, Any]]:
    """
    Validate the thresholds declared on ``scorers`` and build one rule per thresholded scorer.
    """
    name_counts = Counter(scorer.name for scorer in scorers)
    rules = []
    for scorer in scorers:
        for sub_scorer in _sub_scorers(scorer):
            if sub_scorer.quality_threshold is not None:
                raise MlflowException.invalid_parameter_value(
                    f"`quality_threshold` is set on '{sub_scorer.name}', a sub-scorer of the "
                    f"ensemble '{scorer.name}'. Set it on the ensemble scorer instead."
                )

        if (value := getattr(scorer, "quality_threshold", None)) is None:
            continue
        if name_counts[scorer.name] > 1:
            raise MlflowException.invalid_parameter_value(
                f"`quality_threshold` is set on scorer '{scorer.name}', but another scorer "
                "passed to `evaluate` has the same name, so the threshold can't be matched to "
                "a single metric. Give each scorer a unique name."
            )
        threshold = _as_quality_threshold(value)
        bound = "at_least" if threshold.at_least is not None else "at_most"
        rules.append({
            "metricKey": f"{scorer.name}/{threshold.aggregation}",
            "comparator": _COMPARATORS[bound],
            "threshold": getattr(threshold, bound),
            "scorerName": scorer.name,
            "scorerAggregation": threshold.aggregation,
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
            "aggregation": "ALL",
            "missingMetricBehavior": "INCOMPLETE",
            "rules": rules,
        },
        separators=(",", ":"),
    )
    if len(value) > MAX_TAG_VAL_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"The quality thresholds are {len(value)} characters once serialized, over the "
            f"{MAX_TAG_VAL_LENGTH} character limit for a run tag. Set `quality_threshold` on "
            "fewer scorers."
        )
    return value


def warn_on_unmeasured_quality_thresholds(
    rules: list[dict[str, Any]], aggregated_metrics: dict[str, float]
) -> None:
    """
    Warn for each threshold whose metric the run did not produce.

    The UI shows such a run as incomplete rather than passing. This happens when a scorer
    returns non-numeric values or emits Feedback under a different name than its own.
    """
    for rule in rules:
        if rule["metricKey"] not in aggregated_metrics:
            _logger.warning(
                f"The quality threshold on scorer '{rule['scorerName']}' can't be checked: the "
                f"run has no '{rule['metricKey']}' metric. The scorer must return booleans, "
                "numbers, or yes/no values under its own name."
            )
