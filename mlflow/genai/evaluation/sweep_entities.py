"""Result entities for evaluation sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlflow.genai.evaluation.statistics import Interval
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import pandas as pd

    from mlflow.genai.evaluation.entities import EvaluationResult


@experimental(version="3.15.0")
@dataclass
class ScorerInterval:
    """Point estimate and confidence interval for one scorer within a config.

    Args:
        mean: Point estimate — the mean scorer value across repeats.
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        std: Sample standard deviation of the per-repeat means (0 for a single
            repeat, or the row-level std for the bootstrap fallback).
        n_samples: Number of samples backing the interval — the repeat count for
            the t interval, or the pooled trial/row count for Wilson/bootstrap.
        method: Which interval was used: ``"t"``, ``"wilson"``, ``"bootstrap"``,
            or ``"none"`` when too few samples were available.
    """

    mean: float
    ci_low: float
    ci_high: float
    std: float
    n_samples: int
    method: str

    @classmethod
    def from_interval(cls, interval: Interval) -> "ScorerInterval":
        return cls(
            mean=interval.mean,
            ci_low=interval.ci_low,
            ci_high=interval.ci_high,
            std=interval.std,
            n_samples=interval.n_samples,
            method=interval.method,
        )


@experimental(version="3.15.0")
@dataclass
class LatencyStats:
    """Per-row prediction latency for a config, in milliseconds.

    Percentiles are computed over every row of every repeat (``n_rows`` total),
    from each trace's execution duration.
    """

    p50: float
    p90: float
    p99: float
    mean: float
    n_rows: int


@experimental(version="3.15.0")
@dataclass
class SweepConfigResult:
    """Aggregated results for a single swept configuration across all repeats.

    Args:
        name: Config label — the ``predict_fns`` dict key, or the function name
            when a list was passed.
        child_run_ids: MLflow run IDs, one per repeat, all nested under the
            sweep's parent run.
        scorer_intervals: Per-scorer point estimate and confidence interval,
            keyed by scorer name.
        latency: Prediction-latency percentiles across all rows and repeats.
        results: The underlying per-repeat :class:`EvaluationResult` objects, for
            callers who want the raw per-run metrics and result DataFrames.
    """

    name: str
    child_run_ids: list[str] = field(default_factory=list)
    scorer_intervals: dict[str, ScorerInterval] = field(default_factory=dict)
    latency: LatencyStats | None = None
    results: list["EvaluationResult"] = field(default_factory=list)


@experimental(version="3.15.0")
@dataclass
class SweepResult:
    """Result of :func:`mlflow.genai.evaluate_sweep`.

    Args:
        parent_run_id: The parent MLflow run; every config/repeat is a nested
            child run, and flattened summary metrics are logged here.
        configs: Per-config aggregated results, keyed by config name.
        n_repeats: Number of repeats run per config.
    """

    parent_run_id: str
    configs: dict[str, SweepConfigResult] = field(default_factory=dict)
    n_repeats: int = 1

    @property
    def comparison_df(self) -> "pd.DataFrame":
        """Tidy config x scorer table of quality and latency.

        One row per (config, scorer) with the point estimate, confidence
        interval, sample count, interval method, and the config's latency
        percentiles. This is the multi-model quality-vs-latency comparison.
        """
        import pandas as pd

        rows = []
        for config in self.configs.values():
            lat = config.latency
            for scorer_name, interval in config.scorer_intervals.items():
                rows.append({
                    "config": config.name,
                    "scorer": scorer_name,
                    "mean": interval.mean,
                    "ci_low": interval.ci_low,
                    "ci_high": interval.ci_high,
                    "std": interval.std,
                    "n_samples": interval.n_samples,
                    "ci_method": interval.method,
                    "latency_p50_ms": lat.p50 if lat else None,
                    "latency_p90_ms": lat.p90 if lat else None,
                    "latency_p99_ms": lat.p99 if lat else None,
                })
        return pd.DataFrame(rows)

    def best(self, scorer: str, higher_is_better: bool = True) -> str:
        """Name of the config with the best mean for ``scorer``.

        Args:
            scorer: Scorer name to rank configs by.
            higher_is_better: Whether a larger mean is better.

        Returns:
            The winning config's name.
        """
        candidates = {
            name: cfg.scorer_intervals[scorer].mean
            for name, cfg in self.configs.items()
            if scorer in cfg.scorer_intervals
        }
        if not candidates:
            available = sorted({s for c in self.configs.values() for s in c.scorer_intervals})
            raise ValueError(
                f"No config has a result for scorer {scorer!r}. Available scorers: {available}"
            )
        return (max if higher_is_better else min)(candidates, key=candidates.get)

    def __repr__(self) -> str:
        lines = [
            "SweepResult(",
            f"  parent_run_id: {self.parent_run_id}",
            f"  n_repeats: {self.n_repeats}",
            f"  configs ({len(self.configs)}):",
        ]
        for name, cfg in self.configs.items():
            lines.append(f"    {name}:")
            for scorer_name, interval in cfg.scorer_intervals.items():
                lines.append(
                    f"      {scorer_name}: {interval.mean:.3f} "
                    f"[{interval.ci_low:.3f}, {interval.ci_high:.3f}] ({interval.method})"
                )
            if cfg.latency is not None:
                lines.append(
                    f"      latency_ms: p50={cfg.latency.p50:.0f} "
                    f"p90={cfg.latency.p90:.0f} p99={cfg.latency.p99:.0f}"
                )
        lines.append(")")
        return "\n".join(lines)
