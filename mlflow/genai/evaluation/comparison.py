"""Paired statistical comparison of two GenAI evaluation runs."""

import collections
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.metric import Metric
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.aggregation import _cast_assessment_value_to_float
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_BASELINE_SHORT_ID_LENGTH = 8
_MIN_PAIRS = 2
# Below this many discordant pairs the chi-square approximation of McNemar's test is
# unreliable, so the exact binomial test is used instead.
_MCNEMAR_EXACT_MAX_DISCORDANT = 25
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_MAX_CHUNK_ELEMENTS = 1_000_000
_LOGGED_STATS = ("diff", "ci_low", "ci_high", "p_value", "effect_size", "n_paired")

_PairedOn = Literal["eval_request_id", "inputs_hash"]


def _comparison_key(baseline_run_id: str, *parts: str) -> str:
    """
    Build the ``compare/<baseline_short_id>[/<part>...]`` key under which a comparison against
    ``baseline_run_id`` is logged. Metric keys and the artifact path are both derived from it,
    so comparisons of one candidate against several baselines do not overwrite each other.
    """
    return "/".join(["compare", baseline_run_id[:_BASELINE_SHORT_ID_LENGTH], *parts])


@experimental(version="3.17.0")
@dataclass
class ScorerComparison:
    """
    Paired comparison of a single scorer between a candidate and a baseline evaluation run.

    Args:
        scorer: Name of the scorer.
        status: ``"ok"``, or ``"insufficient_pairs"`` when fewer than two rows have a value
            for this scorer in both runs. All statistics are ``None`` in the latter case.
        n_paired: Number of paired rows with a value for this scorer in both runs.
        greater_is_better: Whether a higher value of this scorer is better.
        value_type: ``"binary"`` for pass/fail feedback, ``"numeric"`` otherwise.
        method: Test that produced ``p_value``: ``"mcnemar_exact"``, ``"mcnemar_chi2"`` or
            ``"wilcoxon"``.
        baseline_mean: Mean of the baseline values over the paired rows.
        candidate_mean: Mean of the candidate values over the paired rows.
        diff: Mean paired difference (candidate - baseline).
        ci_low: Lower bound of the percentile bootstrap confidence interval of ``diff``.
        ci_high: Upper bound of the percentile bootstrap confidence interval of ``diff``.
        p_value: Two-sided p-value of ``method``.
        t_test_p_value: Two-sided p-value of the paired t-test. Numeric scorers only. ``NaN``
            when the differences are all the same non-zero value.
        effect_size: Cohen's d_z (mean paired difference / standard deviation of the paired
            differences). ``NaN`` when the differences have no variance.
        ties: Number of pairs whose baseline and candidate values are equal.
    """

    scorer: str
    status: Literal["ok", "insufficient_pairs"]
    n_paired: int
    greater_is_better: bool = True
    value_type: Literal["binary", "numeric"] | None = None
    method: Literal["mcnemar_exact", "mcnemar_chi2", "wilcoxon"] | None = None
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    t_test_p_value: float | None = None
    effect_size: float | None = None
    ties: int | None = None


@experimental(version="3.17.0")
@dataclass
class ComparisonResult:
    """
    Result of :py:func:`mlflow.genai.compare_evaluations`.

    Args:
        candidate_run_id: ID of the candidate evaluation run.
        baseline_run_id: ID of the baseline evaluation run.
        scorers: Per-scorer comparison, keyed by scorer name, for the scorers present in
            both runs.
        paired_rows: One entry per paired row, holding the trace IDs of both runs, how the
            row was paired, and the baseline value, candidate value and delta of each scorer.
        unpaired_baseline_trace_ids: Trace IDs of baseline rows with no match in the candidate.
        unpaired_candidate_trace_ids: Trace IDs of candidate rows with no match in the baseline.
    """

    candidate_run_id: str
    baseline_run_id: str
    scorers: dict[str, ScorerComparison]
    paired_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)
    unpaired_baseline_trace_ids: list[str] = field(default_factory=list)
    unpaired_candidate_trace_ids: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return one row per scorer with its paired statistics."""
        return pd.DataFrame(
            [asdict(comparison) for comparison in self.scorers.values()],
            columns=list(ScorerComparison.__dataclass_fields__),
        )

    def assert_improved(self, scorers: list[str] | None = None, alpha: float = 0.05) -> None:
        """
        Raise ``AssertionError`` unless every listed scorer improved significantly, i.e. moved
        in its better direction with ``p_value < alpha``. A scorer with insufficient pairs is
        treated as not improved.

        Args:
            scorers: Names of the scorers to check. Defaults to all compared scorers, in which
                case ``AssertionError`` is also raised when no scorers were compared.
            alpha: Significance level.
        """
        if scorers is None and not self.scorers:
            raise AssertionError(
                f"No scorers were compared between candidate run {self.candidate_run_id} and "
                f"baseline run {self.baseline_run_id}, so no improvement can be asserted."
            )
        names = list(self.scorers) if scorers is None else scorers
        if unknown := [name for name in names if name not in self.scorers]:
            raise MlflowException.invalid_parameter_value(
                f"Scorers {unknown} were not compared. Compared scorers: {list(self.scorers)}."
            )

        failures = []
        for name in names:
            comparison = self.scorers[name]
            if comparison.status != "ok":
                failures.append(f"{name}: {comparison.status} (n_paired={comparison.n_paired})")
                continue
            direction = 1 if comparison.greater_is_better else -1
            if direction * comparison.diff <= 0 or not comparison.p_value < alpha:
                failures.append(
                    f"{name}: diff={comparison.diff:+.4g} "
                    f"(greater_is_better={comparison.greater_is_better}), "
                    f"p_value={comparison.p_value:.4g}, n_paired={comparison.n_paired}"
                )
        if failures:
            raise AssertionError(
                f"Candidate run {self.candidate_run_id} did not significantly improve over "
                f"baseline run {self.baseline_run_id} at alpha={alpha}:\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )


@dataclass
class _EvalRow:
    trace_id: str
    eval_request_id: str | None
    inputs_hash: str | None
    feedbacks: dict[str, Feedback]


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


def _hash_inputs(trace: Trace) -> str | None:
    root_span = trace.data._get_root_span()
    if root_span is None or root_span.inputs is None:
        return None
    serialized = json.dumps(root_span.inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_eval_rows(run_id: str) -> list[_EvalRow]:
    experiment_id = MlflowClient().get_run(run_id).info.experiment_id
    traces = mlflow.search_traces(locations=[experiment_id], run_id=run_id, return_type="list")
    rows = []
    for trace in traces:
        feedbacks = {
            assessment.name: assessment
            for assessment in trace.info.assessments
            if isinstance(assessment, Feedback)
            and assessment.valid is not False
            and assessment.error is None
            # A trace can be evaluated by several runs. Exclude feedback from other runs.
            and assessment.run_id in (None, run_id)
        }
        rows.append(
            _EvalRow(
                trace_id=trace.info.trace_id,
                eval_request_id=trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID),
                inputs_hash=_hash_inputs(trace),
                feedbacks=feedbacks,
            )
        )
    return rows


def _pair_rows(baseline_rows: list[_EvalRow], candidate_rows: list[_EvalRow]) -> _Pairing:
    """
    Pair rows on the eval request ID, then pair the remaining rows on the hash of their inputs.

    A key shared by several rows of the same run is ambiguous, so those rows are left unpaired.
    """
    pairs = []
    key_names: tuple[_PairedOn, ...] = ("eval_request_id", "inputs_hash")
    for key_name in key_names:

        def unique_by_key(rows: list[_EvalRow]) -> dict[str, _EvalRow]:
            counts = collections.Counter(getattr(row, key_name) for row in rows)
            return {
                key: row
                for row in rows
                if (key := getattr(row, key_name)) is not None and counts[key] == 1
            }

        baseline_by_key = unique_by_key(baseline_rows)
        matched_trace_ids = set()
        for key, candidate in unique_by_key(candidate_rows).items():
            if baseline := baseline_by_key.get(key):
                pairs.append(_RowPair(baseline=baseline, candidate=candidate, paired_on=key_name))
                matched_trace_ids.update((id(baseline), id(candidate)))
        baseline_rows = [row for row in baseline_rows if id(row) not in matched_trace_ids]
        candidate_rows = [row for row in candidate_rows if id(row) not in matched_trace_ids]

    return _Pairing(pairs=pairs, unpaired_baseline=baseline_rows, unpaired_candidate=candidate_rows)


def _is_binary(feedback: Feedback) -> bool:
    # Strings reach the comparison only as yes/no ratings, which are pass/fail like booleans.
    return isinstance(feedback.value, (bool, str))


def _bootstrap_mean_ci(deltas: np.ndarray, confidence_level: float) -> tuple[float, float]:
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    n = len(deltas)
    chunk_size = max(1, _BOOTSTRAP_MAX_CHUNK_ELEMENTS // n)
    means = []
    for start in range(0, _BOOTSTRAP_RESAMPLES, chunk_size):
        size = min(chunk_size, _BOOTSTRAP_RESAMPLES - start)
        means.append(deltas[rng.integers(0, n, size=(size, n))].mean(axis=1))
    tail = (1 - confidence_level) / 2
    low, high = np.quantile(np.concatenate(means), [tail, 1 - tail])
    return float(low), float(high)


def _mcnemar(deltas: np.ndarray) -> tuple[str, float]:
    improved = int(np.sum(deltas > 0))
    regressed = int(np.sum(deltas < 0))
    discordant = improved + regressed
    if discordant == 0:
        return "mcnemar_exact", 1.0
    if discordant < _MCNEMAR_EXACT_MAX_DISCORDANT:
        return "mcnemar_exact", float(stats.binomtest(improved, discordant, 0.5).pvalue)
    statistic = (abs(improved - regressed) - 1) ** 2 / discordant
    return "mcnemar_chi2", float(stats.chi2.sf(statistic, df=1))


def _compare_paired_values(
    scorer: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    binary: bool,
    greater_is_better: bool = True,
    confidence_level: float = 0.95,
) -> ScorerComparison:
    n_paired = len(baseline)
    if n_paired < _MIN_PAIRS:
        return ScorerComparison(
            scorer=scorer,
            status="insufficient_pairs",
            n_paired=n_paired,
            greater_is_better=greater_is_better,
        )

    deltas = candidate - baseline
    diff = float(deltas.mean())
    # Differences that are equal up to floating point noise have no variance to test against.
    constant = np.allclose(deltas, deltas[0], rtol=1e-9, atol=1e-12)
    std = 0.0 if constant else float(deltas.std(ddof=1))
    ci_low, ci_high = _bootstrap_mean_ci(deltas, confidence_level)

    t_test_p_value = None
    if binary:
        method, p_value = _mcnemar(deltas)
    else:
        method = "wilcoxon"
        if constant:
            # Without variance the t statistic is undefined, unless the differences are all
            # zero, which is no evidence of a change.
            t_test_p_value = float("nan") if np.any(deltas != 0) else 1.0
        else:
            t_test_p_value = float(stats.ttest_rel(candidate, baseline).pvalue)
        # The signed-rank test discards zero differences and is undefined when all are zero.
        p_value = float(stats.wilcoxon(deltas).pvalue) if np.any(deltas != 0) else 1.0

    return ScorerComparison(
        scorer=scorer,
        status="ok",
        n_paired=n_paired,
        greater_is_better=greater_is_better,
        value_type="binary" if binary else "numeric",
        method=method,
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        t_test_p_value=t_test_p_value,
        effect_size=diff / std if std > 0 else float("nan"),
        ties=int(np.sum(deltas == 0)),
    )


def _log_comparison(result: ComparisonResult) -> None:
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

    scorers = {
        name: {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in asdict(comparison).items()
        }
        for name, comparison in result.scorers.items()
    }
    client.log_dict(
        result.candidate_run_id,
        {
            "baseline_run_id": result.baseline_run_id,
            "candidate_run_id": result.candidate_run_id,
            "scorers": scorers,
            "unpaired_baseline_trace_ids": result.unpaired_baseline_trace_ids,
            "unpaired_candidate_trace_ids": result.unpaired_candidate_trace_ids,
            "paired_rows": result.paired_rows,
        },
        f"{_comparison_key(result.baseline_run_id)}.json",
    )


@experimental(version="3.17.0")
def compare_evaluations(
    candidate_run_id: str,
    baseline_run_id: str,
    *,
    greater_is_better: dict[str, bool] | None = None,
    confidence_level: float = 0.95,
    log_results: bool = True,
) -> ComparisonResult:
    """
    Compare two :py:func:`mlflow.genai.evaluate` runs with paired statistical tests.

    Rows of the two runs are paired on the evaluation request ID of their traces, falling
    back to a stable hash of the row inputs. Rows that cannot be paired are reported in the
    result and excluded from the statistics. Each scorer that produced feedback in both runs
    is then compared over the paired rows:

    - **Pass/fail feedback** (booleans and yes/no ratings): McNemar's test, exact when there
      are fewer than 25 discordant pairs and chi-square with continuity correction otherwise.
    - **Numeric feedback**: Wilcoxon signed-rank test, with the paired t-test reported
      alongside as ``t_test_p_value``.

    Every scorer also gets the mean paired difference (candidate - baseline), a percentile
    bootstrap confidence interval of that difference (fixed seed, so results are
    reproducible), Cohen's d_z effect size and the number of ties. A scorer with fewer than
    two paired values is reported with status ``"insufficient_pairs"``. If no scorer produced
    valid feedback in both runs, a warning is logged and the result has no scorers.

    No correction for multiple comparisons is applied across scorers.

    Args:
        candidate_run_id: ID of the evaluation run being assessed.
        baseline_run_id: ID of the evaluation run to compare against.
        greater_is_better: Per-scorer direction overrides, keyed by scorer name. Scorers not
            listed are assumed to be better when higher. Only affects
            :py:meth:`ComparisonResult.assert_improved`; ``diff`` is always
            candidate - baseline.
        confidence_level: Confidence level of the bootstrap confidence interval.
        log_results: If True, log the comparison to the candidate run: metrics
            ``compare/<baseline_short_id>/<scorer>/{diff,ci_low,ci_high,p_value,effect_size,
            n_paired}`` and one ``compare/<baseline_short_id>.json`` artifact containing the
            full baseline run ID and the per-row paired deltas. ``<baseline_short_id>`` is the
            first 8 characters of the baseline run ID.

    Returns:
        A :py:class:`ComparisonResult <mlflow.genai.evaluation.comparison.ComparisonResult>`.

    Example:

    .. code-block:: python

        import mlflow

        baseline = mlflow.genai.evaluate(data=data, predict_fn=agent_v1, scorers=scorers)
        candidate = mlflow.genai.evaluate(data=data, predict_fn=agent_v2, scorers=scorers)

        result = mlflow.genai.compare_evaluations(
            candidate_run_id=candidate.run_id,
            baseline_run_id=baseline.run_id,
        )
        print(result.summary())
        result.assert_improved(scorers=["correctness"], alpha=0.05)
    """
    if candidate_run_id == baseline_run_id:
        raise MlflowException.invalid_parameter_value(
            "`candidate_run_id` and `baseline_run_id` must be different runs."
        )
    if not 0 < confidence_level < 1:
        raise MlflowException.invalid_parameter_value(
            f"`confidence_level` must be between 0 and 1, got {confidence_level}."
        )
    greater_is_better = greater_is_better or {}

    pairing = _pair_rows(_load_eval_rows(baseline_run_id), _load_eval_rows(candidate_run_id))
    if pairing.unpaired_baseline or pairing.unpaired_candidate:
        _logger.warning(
            f"{len(pairing.unpaired_baseline)} baseline row(s) and "
            f"{len(pairing.unpaired_candidate)} candidate row(s) could not be paired and are "
            "excluded from the comparison."
        )

    def scorer_names(rows: list[_EvalRow]) -> set[str]:
        return {name for row in rows for name in row.feedbacks}

    baseline_scorers = scorer_names([
        *pairing.unpaired_baseline,
        *(p.baseline for p in pairing.pairs),
    ])
    candidate_scorers = scorer_names([
        *pairing.unpaired_candidate,
        *(p.candidate for p in pairing.pairs),
    ])
    common_scorers = sorted(baseline_scorers & candidate_scorers)
    if not common_scorers:
        _logger.warning(
            "No scorers were compared because the baseline and candidate runs have no scorer "
            "with valid feedback in common. Scorers with valid feedback in the baseline run: "
            f"{sorted(baseline_scorers)}. Scorers with valid feedback in the candidate run: "
            f"{sorted(candidate_scorers)}."
        )

    paired_rows = [
        {
            "baseline_trace_id": pair.baseline.trace_id,
            "candidate_trace_id": pair.candidate.trace_id,
            "paired_on": pair.paired_on,
            "scores": {},
        }
        for pair in pairing.pairs
    ]
    comparisons = {}
    for name in common_scorers:
        baseline_values = []
        candidate_values = []
        binary = True
        for pair, paired_row in zip(pairing.pairs, paired_rows):
            baseline_feedback = pair.baseline.feedbacks.get(name)
            candidate_feedback = pair.candidate.feedbacks.get(name)
            if baseline_feedback is None or candidate_feedback is None:
                continue
            baseline_value = _cast_assessment_value_to_float(baseline_feedback)
            candidate_value = _cast_assessment_value_to_float(candidate_feedback)
            if baseline_value is None or candidate_value is None:
                continue
            binary = binary and _is_binary(baseline_feedback) and _is_binary(candidate_feedback)
            baseline_values.append(baseline_value)
            candidate_values.append(candidate_value)
            paired_row["scores"][name] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
            }

        comparisons[name] = _compare_paired_values(
            name,
            np.array(baseline_values, dtype=float),
            np.array(candidate_values, dtype=float),
            binary=binary,
            greater_is_better=greater_is_better.get(name, True),
            confidence_level=confidence_level,
        )

    result = ComparisonResult(
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
        scorers=comparisons,
        paired_rows=paired_rows,
        unpaired_baseline_trace_ids=[row.trace_id for row in pairing.unpaired_baseline],
        unpaired_candidate_trace_ids=[row.trace_id for row in pairing.unpaired_candidate],
    )
    if log_results:
        _log_comparison(result)
    return result
