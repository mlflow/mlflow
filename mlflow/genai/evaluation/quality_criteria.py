"""Quality criteria passed to ``evaluate``, recorded on the evaluation run as a tag."""

import json
import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

# NB: The payload follows the criteria JSON the evaluation runs UI already reads.
_SCHEMA_VERSION = 1
_COMPARATORS = {">=": "GTE", "<=": "LTE"}
_EXAMPLE = "correctness >= 0.9 AND `latency/p90` <= 2"
_CONDITION = re.compile(
    r'\s*(?:`(?P<quoted>[^`]+)`|"(?P<double_quoted>[^"]+)"|(?P<bare>[A-Za-z_][\w.]*))'
    r"\s*(?P<comparator>>=|<=|!=|=|>|<)"
    r"\s*(?P<threshold>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
)
_AND = re.compile(r"(?<=\s)AND\s", re.IGNORECASE)


@dataclass(frozen=True)
class QualityCriteriaRule:
    scorer_name: str
    aggregation: str
    comparator: str
    threshold: float

    @property
    def metric_key(self) -> str:
        return f"{self.scorer_name}/{self.aggregation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metricKey": self.metric_key,
            "comparator": _COMPARATORS[self.comparator],
            "threshold": self.threshold,
        }


def parse_quality_criteria(
    scorers: list[Scorer], quality_criteria: str | None
) -> list[QualityCriteriaRule]:
    """
    Parse ``quality_criteria`` into one rule per condition and validate it against ``scorers``.

    The syntax is like trace search filters: conditions joined by ``AND``, each of the
    form ``<metric> >= <number>`` or ``<metric> <= <number>``. ``<metric>`` is a scorer name,
    which checks the scorer's mean, or ``<scorer name>/<aggregation>`` in backticks.
    """
    if quality_criteria is None:
        return []
    if not isinstance(quality_criteria, str):
        raise MlflowException.invalid_parameter_value(
            "`quality_criteria` must be a string such as "
            f"{_EXAMPLE!r}, got {type(quality_criteria).__name__}."
        )
    if not quality_criteria.strip():
        return []

    # NB: `getattr`, because legacy Databricks `Metric` objects aren't scorers.
    name_counts = Counter(getattr(scorer, "name", None) for scorer in scorers)
    rules = []
    pos = 0
    while True:
        if not (match := _CONDITION.match(quality_criteria, pos)):
            raise _invalid_syntax(quality_criteria, pos)
        rule = _parse_condition(match, name_counts)
        if any(r.metric_key == rule.metric_key for r in rules):
            raise MlflowException.invalid_parameter_value(
                f"Invalid quality criteria: more than one condition on {rule.metric_key!r}. "
                "Set one condition per metric."
            )
        rules.append(rule)
        if (pos := match.end()) == len(quality_criteria):
            return rules
        if not (match := _AND.match(quality_criteria, pos)):
            raise _invalid_syntax(quality_criteria, pos)
        pos = match.end()


def _parse_condition(match: re.Match[str], name_counts: Counter) -> QualityCriteriaRule:
    # Lazy import: `aggregation` needs numpy, which mlflow-skinny doesn't install.
    from mlflow.genai.scorers.aggregation import _AGGREGATE_FUNCTIONS

    condition = match.group().strip()
    if not math.isfinite(threshold := float(match["threshold"])):
        raise MlflowException.invalid_parameter_value(
            f"Invalid condition {condition!r}: the threshold must be a finite number."
        )
    if (comparator := match["comparator"]) not in _COMPARATORS:
        raise MlflowException.invalid_parameter_value(
            f"Invalid condition {condition!r}: only `>=` and `<=` are supported."
        )

    key = match["quoted"] or match["double_quoted"] or match["bare"]
    scorer_name = key
    aggregation = "mean"
    if name_counts[key] == 0 and "/" in key:
        prefix, suffix = key.rsplit("/", 1)
        if name_counts[prefix] > 0 and suffix in _AGGREGATE_FUNCTIONS:
            scorer_name = prefix
            aggregation = suffix

    if name_counts[scorer_name] == 0:
        scorer_names = ", ".join(repr(n) for n in sorted(n for n in name_counts if n))
        raise MlflowException.invalid_parameter_value(
            f"The condition on {key!r} doesn't match any scorer passed to `evaluate`. "
            f"Use a scorer name to check its mean, or `<scorer name>/<aggregation>` in "
            f"backticks, where <aggregation> is one of {', '.join(_AGGREGATE_FUNCTIONS)}. "
            f"Scorer names: {scorer_names}."
        )
    if name_counts[scorer_name] > 1:
        raise MlflowException.invalid_parameter_value(
            f"The condition on {key!r} matches more than one scorer passed to "
            "`evaluate`, so it can't be matched to a single metric. Give each scorer a "
            "unique name."
        )
    return QualityCriteriaRule(scorer_name, aggregation, comparator, threshold)


def _invalid_syntax(quality_criteria: str, pos: int) -> MlflowException:
    return MlflowException.invalid_parameter_value(
        f"Invalid quality criteria {quality_criteria!r} at {quality_criteria[pos:]!r}. "
        f"Expected conditions joined by AND, such as {_EXAMPLE!r}. Wrap names that contain "
        "spaces or symbols in backticks."
    )


def get_extra_aggregations(rules: list[QualityCriteriaRule]) -> dict[str, list[str]]:
    """Map each scorer name to the aggregations its conditions are checked against."""
    aggregations = defaultdict(list)
    for rule in rules:
        aggregations[rule.scorer_name].append(rule.aggregation)
    return dict(aggregations)


def build_quality_criteria_tag(rules: list[QualityCriteriaRule]) -> str | None:
    """
    Serialize the rules from ``parse_quality_criteria`` for the run tag.

    Returns:
        The tag value, or None when there are no rules.
    """
    if not rules:
        return None

    value = json.dumps(
        {
            "schemaVersion": _SCHEMA_VERSION,
            "revision": str(uuid.uuid4()),
            "rules": [rule.to_dict() for rule in rules],
        },
        separators=(",", ":"),
    )
    if len(value) > MAX_TAG_VAL_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"The quality criteria are {len(value)} characters once serialized, over the "
            f"{MAX_TAG_VAL_LENGTH} character limit for a run tag. Set fewer conditions."
        )
    return value


def warn_on_unmeasured_quality_criteria(
    rules: list[QualityCriteriaRule], aggregated_metrics: dict[str, float]
) -> None:
    """
    Warn for each condition whose metric this evaluation did not produce.

    This happens when a scorer returns non-numeric values, emits Feedback under a different
    name than its own, or fails on every row.
    """
    for rule in rules:
        if rule.metric_key not in aggregated_metrics:
            _logger.warning(
                f"The condition on scorer '{rule.scorer_name}' can't be checked: this "
                f"evaluation produced no '{rule.metric_key}' metric. The scorer must return "
                "booleans, numbers, or yes/no values under its own name. If the run already has "
                "this metric from an earlier evaluation, the condition is checked against "
                "that value."
            )
