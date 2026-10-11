"""Paired statistical comparison of two GenAI evaluation runs."""

import collections
import hashlib
import inspect
import json
import logging
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.metric import Metric
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.time import get_current_time_millis

# This module is imported by `import mlflow`, which must work on `mlflow-skinny`, so the data
# science libraries are only imported inside the functions that use them.
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

_logger = logging.getLogger(__name__)

_BASELINE_SHORT_ID_LENGTH = 8
# The smallest n at which the exact two-sided signed-rank p-value can fall below 0.05
# (2 / 2**6 = 0.03125). Below it no result could ever be significant.
_MIN_PAIRS = 6
# Below this many pairs the result is reported but a warning is logged.
_SMALL_SAMPLE_PAIRS = 10
# Below this many discordant pairs the chi-square approximation of McNemar's test is
# unreliable, so the exact binomial test is used instead.
_MCNEMAR_EXACT_MAX_DISCORDANT = 25
# Up to this many non-zero differences the null distribution of the signed-rank statistic is
# enumerated exactly; above it the normal approximation is used.
_WILCOXON_EXACT_MAX_NONZERO = 50
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 10_000
# Bounds the memory of one bootstrap batch (resamples x pairs).
_BOOTSTRAP_MAX_BATCH_ELEMENTS = 1_000_000
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


@experimental(version="3.18.0")
@dataclass
class ScorerComparison:
    """
    Paired comparison of a single scorer between a candidate and a baseline evaluation run.

    Args:
        scorer: Name of the scorer.
        status: ``"ok"``, or ``"insufficient_pairs"`` when fewer than six rows have a value
            for this scorer in both runs. All statistics are ``None`` in the latter case.
        n_paired: Number of paired rows with a value for this scorer in both runs.
        greater_is_better: Whether a higher value of this scorer is better.
        value_type: ``"binary"`` when every feedback of the scorer is a boolean or a yes/no
            rating, ``"numeric"`` otherwise. Describes the feedback, not the test: that is
            chosen from the paired values, see ``method``.
        method: Test that produced ``p_value``. ``"mcnemar_exact"`` or ``"mcnemar_chi2"``
            when every paired value is 0 or 1 (exact binomial below 25 discordant pairs),
            ``"wilcoxon_exact"`` or ``"wilcoxon_asymptotic"`` otherwise (exact enumeration up
            to 50 non-zero differences, normal approximation above).
        baseline_mean: Mean of the baseline values over the paired rows.
        candidate_mean: Mean of the candidate values over the paired rows.
        diff: Mean paired difference (candidate - baseline).
        ci_low: Lower bound of the bias-corrected and accelerated (BCa) bootstrap confidence
            interval of ``diff``.
        ci_high: Upper bound of the BCa bootstrap confidence interval of ``diff``.
        p_value: Two-sided p-value of ``method``.
        t_test_p_value: Two-sided p-value of the paired t-test. Numeric scorers only. ``NaN``
            when the differences are all the same non-zero value.
        effect_size: Cohen's d_z (mean paired difference / standard deviation of the paired
            differences). ``NaN`` when the differences have no variance.
        ties: Number of pairs whose baseline and candidate values are equal.
        n_nonzero: Number of pairs whose baseline and candidate values differ, i.e. the pairs
            the test is computed on: the signed-rank test drops zero differences and McNemar's
            test only counts discordant pairs. ``diff`` and its confidence interval use all
            ``n_paired`` pairs.
    """

    scorer: str
    status: Literal["ok", "insufficient_pairs"]
    n_paired: int
    greater_is_better: bool = True
    value_type: Literal["binary", "numeric"] | None = None
    method: (
        Literal["mcnemar_exact", "mcnemar_chi2", "wilcoxon_exact", "wilcoxon_asymptotic"] | None
    ) = None
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    t_test_p_value: float | None = None
    effect_size: float | None = None
    ties: int | None = None
    n_nonzero: int | None = None


@experimental(version="3.18.0")
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

    def summary(self) -> "pd.DataFrame":
        """Return one row per scorer with its paired statistics."""
        import pandas as pd

        return pd.DataFrame(
            [asdict(comparison) for comparison in self.scorers.values()],
            columns=list(ScorerComparison.__dataclass_fields__),
        )

    def assert_improved(self, scorers: list[str] | None = None, alpha: float = 0.05) -> None:
        """
        Raise ``AssertionError`` unless every listed scorer improved significantly, i.e. moved
        in its better direction with ``p_value < alpha``. The confidence interval is not
        consulted. A scorer with insufficient pairs is treated as not improved.

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
    # `mlflow.genai.scorers.aggregation` imports numpy at module level.
    from mlflow.genai.scorers.aggregation import _cast_assessment_value_to_float

    values = [
        value
        for feedback in feedbacks
        if (value := _cast_assessment_value_to_float(feedback)) is not None
    ]
    return sum(values) / len(values) if values else None


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


def _is_pass_fail(feedback: Feedback) -> bool:
    # Strings reach the comparison only as yes/no ratings, which are pass/fail like booleans.
    return isinstance(feedback.value, (bool, str))


def _bootstrap_mean_ci(deltas: "np.ndarray", confidence_level: float) -> tuple[float, float]:
    """
    BCa bootstrap confidence interval of the mean of ``deltas``, which must not be constant
    (the BCa acceleration is undefined then). Falls back to the percentile interval in the
    rare case that BCa still yields a non-finite bound.
    """
    import numpy as np
    from scipy import stats

    # scipy 1.15 renamed the seed argument from `random_state` to `rng`.
    seed_kwarg = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"

    def bootstrap(method: str) -> tuple[float, float]:
        with warnings.catch_warnings():
            # A degenerate BCa interval is handled below. Its warnings add nothing.
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", stats.DegenerateDataWarning)
            interval = stats.bootstrap(
                (deltas,),
                np.mean,
                method=method,
                confidence_level=confidence_level,
                n_resamples=_BOOTSTRAP_RESAMPLES,
                batch=max(1, _BOOTSTRAP_MAX_BATCH_ELEMENTS // len(deltas)),
                **{seed_kwarg: np.random.default_rng(_BOOTSTRAP_SEED)},
            ).confidence_interval
        return float(interval.low), float(interval.high)

    low, high = bootstrap("BCa")
    if not (math.isfinite(low) and math.isfinite(high)):
        _logger.warning(
            "The BCa bootstrap confidence interval is undefined for this sample; reporting "
            "the percentile bootstrap interval instead."
        )
        low, high = bootstrap("percentile")
    return low, high


def _mcnemar(deltas: "np.ndarray") -> tuple[str, float]:
    import numpy as np
    from scipy import stats

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
    baseline: "np.ndarray",
    candidate: "np.ndarray",
    *,
    value_type: Literal["binary", "numeric"],
    greater_is_better: bool = True,
    confidence_level: float = 0.95,
) -> ScorerComparison:
    import numpy as np
    from scipy import stats

    n_paired = len(baseline)
    if n_paired < _MIN_PAIRS:
        return ScorerComparison(
            scorer=scorer,
            status="insufficient_pairs",
            n_paired=n_paired,
            greater_is_better=greater_is_better,
        )
    if n_paired < _SMALL_SAMPLE_PAIRS:
        _logger.warning(
            f"Scorer {scorer!r} has only {n_paired} paired rows. With fewer than "
            f"{_SMALL_SAMPLE_PAIRS} pairs the tests have little power and the confidence "
            "interval is unreliable; evaluate more rows before trusting the result."
        )

    # The test is chosen from the paired values, not from the Python type of the feedback:
    # McNemar's test applies iff every value of both runs is 0 or 1. An ``int(x > 0.5)`` or
    # ``1.0``/``0.0`` scorer is pass/fail in substance, whereas a boolean scorer whose
    # per-trace averages are fractional is not.
    binary = bool(np.isin(np.concatenate([baseline, candidate]), (0.0, 1.0)).all())
    deltas = candidate - baseline
    diff = float(deltas.mean())
    # Differences that are equal up to floating point noise have no variance to test against.
    constant = np.allclose(deltas, deltas[0], rtol=1e-9, atol=1e-12)
    std = 0.0 if constant else float(deltas.std(ddof=1))
    # Resampling a constant can only return it, and the BCa acceleration is undefined there.
    ci_low, ci_high = (diff, diff) if constant else _bootstrap_mean_ci(deltas, confidence_level)

    n_nonzero = int(np.sum(deltas != 0))
    t_test_p_value = None
    if binary:
        method, p_value = _mcnemar(deltas)
    else:
        if constant:
            # Without variance the t statistic is undefined, unless the differences are all
            # zero, which is no evidence of a change.
            t_test_p_value = float("nan") if n_nonzero else 1.0
        else:
            t_test_p_value = float(stats.ttest_rel(candidate, baseline).pvalue)
        # scipy's ``method="auto"`` switches between the exact distribution, a permutation test
        # (randomized above 13 pairs) and the normal approximation depending on the sample
        # size, ties and the scipy version, so one extra row could move the p-value to another
        # regime. The choice is pinned on the number of non-zero differences instead. The exact
        # distribution is conservative when absolute differences tie. scipy >= 1.13 calls the
        # normal approximation "asymptotic" and keeps "approx" as an alias; older versions
        # only know "approx".
        exact = n_nonzero <= _WILCOXON_EXACT_MAX_NONZERO
        method = "wilcoxon_exact" if exact else "wilcoxon_asymptotic"
        # The signed-rank test discards zero differences and is undefined when all are zero.
        p_value = (
            float(stats.wilcoxon(deltas, method="exact" if exact else "approx").pvalue)
            if n_nonzero
            else 1.0
        )

    return ScorerComparison(
        scorer=scorer,
        status="ok",
        n_paired=n_paired,
        greater_is_better=greater_is_better,
        value_type=value_type,
        method=method,
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        t_test_p_value=t_test_p_value,
        effect_size=diff / std if std > 0 else float("nan"),
        ties=n_paired - n_nonzero,
        n_nonzero=n_nonzero,
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


@experimental(version="3.18.0")
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
    result and excluded from the statistics. Several feedbacks of one name on a trace (e.g.
    from a scorer returning a list of feedbacks) are averaged per trace, as
    :py:func:`mlflow.genai.evaluate` averages them into ``<scorer>/mean``. Each scorer that
    produced feedback in both runs is then compared over the paired rows, with the test
    chosen from the paired values rather than from the type the scorer returned:

    - **Every paired value is 0 or 1** (booleans, yes/no ratings, but also ``int(x > 0.5)``
      or ``1.0``/``0.0`` scorers): McNemar's test, exact when there are fewer than 25
      discordant pairs and chi-square with continuity correction otherwise.
    - **Otherwise** (including pass/fail scorers whose per-trace averages are fractional):
      Wilcoxon signed-rank test, exact up to 50 non-zero differences and with the normal
      approximation above, with the paired t-test reported alongside as ``t_test_p_value``.

    Every scorer also gets the mean paired difference (candidate - baseline), a bias-corrected
    and accelerated (BCa) bootstrap confidence interval of that difference (fixed seed, so
    results are reproducible), Cohen's d_z effect size and the numbers of tied and non-zero
    differences. A scorer with
    fewer than six paired values, the smallest sample at which a result can be significant at
    0.05, is reported with status ``"insufficient_pairs"``, and a warning is logged for
    scorers with fewer than ten. If no scorer produced valid feedback in both runs, a warning
    is logged and the result has no scorers.

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
    import numpy as np

    greater_is_better = greater_is_better or {}

    pairing = _pair_rows(_load_eval_rows(baseline_run_id), _load_eval_rows(candidate_run_id))
    _warn_unpaired_rows(pairing)

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
        pass_fail = True
        for pair, paired_row in zip(pairing.pairs, paired_rows):
            baseline_feedbacks = pair.baseline.feedbacks.get(name, [])
            candidate_feedbacks = pair.candidate.feedbacks.get(name, [])
            baseline_value = _mean_feedback_value(baseline_feedbacks)
            candidate_value = _mean_feedback_value(candidate_feedbacks)
            if baseline_value is None or candidate_value is None:
                continue
            pass_fail = pass_fail and all(
                _is_pass_fail(feedback) for feedback in (*baseline_feedbacks, *candidate_feedbacks)
            )
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
            value_type="binary" if pass_fail else "numeric",
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
