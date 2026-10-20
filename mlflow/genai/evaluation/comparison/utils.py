"""Loading, pairing and logging of the rows of two evaluation runs."""

import collections
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.metric import Metric
from mlflow.entities.trace import Trace
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracking.client import MlflowClient
from mlflow.utils.time import get_current_time_millis

if TYPE_CHECKING:
    from mlflow.genai.evaluation.comparison.comparison import ComparisonResult

_logger = logging.getLogger(__name__)

_BASELINE_SHORT_ID_LENGTH = 8
_LOGGED_STATS = ("diff", "ci_low", "ci_high", "p_value", "effect_size", "n_paired")

_PairedOn = Literal["eval_request_id", "inputs_hash"]
# ``str()`` of an object without a JSON form, e.g. ``<Query object at 0x10f3b2d40>``.
_MEMORY_ADDRESS_PATTERN = re.compile(r" at 0x[0-9a-fA-F]+")


def _comparison_key(baseline_run_id: str, *parts: str) -> str:
    """
    Build the ``compare/<baseline_short_id>[/<part>...]`` key under which a comparison against
    ``baseline_run_id`` is logged. Metric keys and the artifact path are both derived from it,
    so comparisons of one candidate against several baselines do not overwrite each other.
    """
    return "/".join(["compare", baseline_run_id[:_BASELINE_SHORT_ID_LENGTH], *parts])


@dataclass
class _EvalRow:
    trace_id: str
    eval_request_id: str | None
    inputs_hash: str | None
    # Whether ``inputs_hash`` differs between runs because an input was serialized by its
    # memory address, see ``_hash_inputs``.
    inputs_hash_unstable: bool
    # A trace can hold several feedbacks of one name, e.g. when a scorer returns a list of
    # feedbacks. They are averaged per trace before pairing, see ``_mean_feedback_value``.
    feedbacks: dict[str, list[Feedback]]


@dataclass
class _RowPair:
    baseline: _EvalRow
    candidate: _EvalRow
    paired_on: _PairedOn


@dataclass
class _Pairing:
    pairs: list[_RowPair]
    unpaired_baseline: list[_EvalRow]
    unpaired_candidate: list[_EvalRow]
    # Trace IDs of the unpaired rows that were left unpaired because their key is shared by
    # another row of the same run. The other unpaired rows have no match in the other run.
    duplicate_key_trace_ids: set[str]


def _hash_inputs(trace: Trace) -> tuple[str | None, bool]:
    """
    Hash of the root span inputs, and whether it is unstable across runs because an input
    without a JSON form was serialized by its memory address (``<Foo object at 0x...>``).
    """
    root_span = trace.data._get_root_span()
    if root_span is None or root_span.inputs is None:
        return None, False
    serialized = json.dumps(root_span.inputs, sort_keys=True, default=str)
    unstable = _MEMORY_ADDRESS_PATTERN.search(serialized) is not None
    return hashlib.sha256(serialized.encode()).hexdigest(), unstable


def _load_eval_rows(run_id: str) -> list[_EvalRow]:
    experiment_id = MlflowClient().get_run(run_id).info.experiment_id
    traces = mlflow.search_traces(locations=[experiment_id], run_id=run_id, return_type="list")
    rows = []
    for trace in traces:
        feedbacks = collections.defaultdict(list)
        for assessment in trace.info.assessments:
            if (
                isinstance(assessment, Feedback)
                and assessment.valid is not False
                and assessment.error is None
                # A trace can be evaluated by several runs. Exclude feedback from other runs.
                and assessment.run_id in (None, run_id)
            ):
                feedbacks[assessment.name].append(assessment)
        inputs_hash, inputs_hash_unstable = _hash_inputs(trace)
        rows.append(
            _EvalRow(
                trace_id=trace.info.trace_id,
                eval_request_id=trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID),
                inputs_hash=inputs_hash,
                inputs_hash_unstable=inputs_hash_unstable,
                feedbacks=dict(feedbacks),
            )
        )
    return rows


def _mean_feedback_value(feedbacks: list[Feedback]) -> float | None:
    """
    Average the values of the same-name feedbacks of one trace into the row's value for the
    scorer. :py:func:`mlflow.genai.evaluate` pools all of a run's same-name feedback values
    into its ``<scorer>/mean`` metric, so the per-trace mean is the value that metric is made
    of. Returns ``None`` when no feedback has a value that can be cast to a float.
    """
    # `mlflow.genai.scorers.aggregation` imports numpy at module level, which `mlflow-skinny`
    # does not have.
    from mlflow.genai.scorers.aggregation import _cast_assessment_value_to_float

    values = [
        value
        for feedback in feedbacks
        if (value := _cast_assessment_value_to_float(feedback)) is not None
    ]
    return sum(values) / len(values) if values else None


def _is_pass_fail(feedback: Feedback) -> bool:
    # Strings reach the comparison only as yes/no ratings, which are pass/fail like booleans.
    return isinstance(feedback.value, (bool, str))


def _pair_rows(baseline_rows: list[_EvalRow], candidate_rows: list[_EvalRow]) -> _Pairing:
    """
    Pair rows on the eval request ID, then pair the remaining rows on the hash of their inputs.

    A key shared by several rows of the same run is ambiguous, so those rows are left unpaired.
    """
    pairs = []
    duplicate_key_trace_ids = set()
    key_names: tuple[_PairedOn, ...] = ("eval_request_id", "inputs_hash")
    for key_name in key_names:

        def unique_by_key(rows: list[_EvalRow]) -> dict[str, _EvalRow]:
            counts = collections.Counter(getattr(row, key_name) for row in rows)
            unique = {}
            for row in rows:
                if (key := getattr(row, key_name)) is None:
                    continue
                if counts[key] == 1:
                    unique[key] = row
                else:
                    duplicate_key_trace_ids.add(row.trace_id)
            return unique

        baseline_by_key = unique_by_key(baseline_rows)
        matched_trace_ids = set()
        for key, candidate in unique_by_key(candidate_rows).items():
            if baseline := baseline_by_key.get(key):
                pairs.append(_RowPair(baseline=baseline, candidate=candidate, paired_on=key_name))
                matched_trace_ids.update((id(baseline), id(candidate)))
        baseline_rows = [row for row in baseline_rows if id(row) not in matched_trace_ids]
        candidate_rows = [row for row in candidate_rows if id(row) not in matched_trace_ids]

    unpaired_trace_ids = {row.trace_id for row in (*baseline_rows, *candidate_rows)}
    return _Pairing(
        pairs=pairs,
        unpaired_baseline=baseline_rows,
        unpaired_candidate=candidate_rows,
        # A row with a duplicate eval request ID can still have been paired on its inputs.
        duplicate_key_trace_ids=duplicate_key_trace_ids & unpaired_trace_ids,
    )


def _warn_unpaired_rows(pairing: _Pairing) -> None:
    unpaired = [*pairing.unpaired_baseline, *pairing.unpaired_candidate]
    if not unpaired:
        return
    no_match = [row for row in unpaired if row.trace_id not in pairing.duplicate_key_trace_ids]
    n_duplicate = len(unpaired) - len(no_match)
    reasons = []
    if no_match:
        reasons.append(
            f"{len(no_match)} row(s) have no match in the other run; evaluate both runs on the "
            "same rows, or inspect `unpaired_baseline_trace_ids` and "
            "`unpaired_candidate_trace_ids` for the rows that differ."
        )
    if n_unstable := sum(row.inputs_hash_unstable for row in no_match):
        reasons.append(
            f"{n_unstable} of the unmatched row(s) have inputs that were serialized by their "
            "memory address (e.g. '<Query object at 0x...>'), which differs between runs, so "
            "they cannot be paired on their inputs; make the inputs JSON-serializable, or "
            "evaluate a static dataset so that rows pair on their eval request ID."
        )
    if n_duplicate:
        reasons.append(
            f"{n_duplicate} row(s) share their eval request ID or inputs with another row of "
            "the same run, which makes their pairing ambiguous; remove duplicate rows from the "
            "evaluation dataset."
        )
    _logger.warning(
        f"{len(pairing.unpaired_baseline)} baseline row(s) and "
        f"{len(pairing.unpaired_candidate)} candidate row(s) could not be paired and are "
        "excluded from the comparison. " + " ".join(reasons)
    )


def _log_comparison(result: "ComparisonResult") -> None:
    client = MlflowClient()
    timestamp = get_current_time_millis()
    metrics = []
    for name, comparison in result.scorers.items():
        for stat in _LOGGED_STATS:
            value = getattr(comparison, stat)
            if value is not None and math.isfinite(value):
                key = _comparison_key(result.baseline_run_id, name, stat)
                metrics.append(Metric(key=key, value=float(value), timestamp=timestamp, step=0))
    if metrics:
        client.log_batch(result.candidate_run_id, metrics=metrics)
    # The metric keys only carry the short baseline ID. The artifact maps it back to the run.
    client.log_dict(
        result.candidate_run_id,
        {"baseline_run_id": result.baseline_run_id},
        f"{_comparison_key(result.baseline_run_id)}.json",
    )
