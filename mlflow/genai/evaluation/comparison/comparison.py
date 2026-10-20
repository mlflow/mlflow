"""Public API of the paired comparison of two GenAI evaluation runs."""

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.comparison.paired_comparison import _compare_paired_values
from mlflow.genai.evaluation.comparison.utils import (
    _EvalRow,
    _is_pass_fail,
    _load_eval_rows,
    _log_comparison,
    _mean_feedback_value,
    _pair_rows,
    _warn_unpaired_rows,
)
from mlflow.utils.annotations import experimental

# This module is imported by `import mlflow`, which must work on `mlflow-skinny`, so the data
# science libraries are only imported inside the functions that use them.
if TYPE_CHECKING:
    import pandas as pd

_logger = logging.getLogger(__name__)


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
            ``ComparisonResult.assert_improved()``; ``diff`` is always
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
