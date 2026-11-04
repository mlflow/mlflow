"""Result types of the paired comparison of two GenAI evaluation runs."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental

# This module is imported by `import mlflow`, which must work on `mlflow-skinny`, so pandas is
# only imported inside the method that uses it.
if TYPE_CHECKING:
    import pandas as pd


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
                case ``AssertionError`` is also raised when no scorers were compared. An
                empty list is rejected, since it would check nothing.
            alpha: Significance level.
        """
        if scorers is not None and not scorers:
            raise MlflowException.invalid_parameter_value(
                "`scorers` must not be empty. Pass `None` to check every compared scorer."
            )
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
