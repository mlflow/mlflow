"""Evaluate multiple configurations with repeated runs for confidence intervals.

:func:`evaluate_sweep` runs :func:`mlflow.genai.evaluate` across a grid of
``|predict_fns| x n_repeats`` cells, comparing configs on quality and latency
with a confidence interval per scorer.

Each cell is a nested MLflow run under one parent run, reusing the existing
single-run ``evaluate`` path (no harness changes). Cells run sequentially:
MLflow's fluent run stack and autologging are process-global, so concurrent
nested runs would corrupt run/trace association, and the harness already
parallelizes rows within a cell. Confidence intervals are chosen automatically
in :mod:`mlflow.genai.evaluation.statistics`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.base import evaluate
from mlflow.genai.evaluation.statistics import compute_scorer_interval
from mlflow.genai.evaluation.sweep_entities import (
    LatencyStats,
    ScorerInterval,
    SweepConfigResult,
    SweepResult,
)
from mlflow.genai.scorers import Scorer
from mlflow.utils.annotations import experimental
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes

_logger = logging.getLogger(__name__)

# Tag applied to the parent run so it can be distinguished from a single evaluate run.
MLFLOW_RUN_TYPE_GENAI_EVALUATE_SWEEP = "genai_evaluate_sweep"

# Suffix of the result_df columns that hold a scorer's per-row value.
_VALUE_COLUMN_SUFFIX = "/value"


@experimental(version="3.15.0")
def evaluate_sweep(
    data: "EvaluationDatasetTypes",
    scorers: list[Scorer],
    predict_fns: dict[str, Callable[..., Any]] | list[Callable[..., Any]],
    *,
    n_repeats: int = 3,
    predict_once: bool = False,
) -> SweepResult:
    """Evaluate multiple configurations repeatedly and compare them with confidence intervals.

    This runs :func:`mlflow.genai.evaluate` once per (config, repeat) cell over a
    grid of ``len(predict_fns) x n_repeats`` cells against the same ``data`` and
    ``scorers``. Repeating each config yields a confidence interval per scorer,
    and evaluating multiple configs produces a side-by-side quality and latency
    comparison — useful for choosing between models/agents and for quantifying
    run-to-run variance.

    Each cell is a nested MLflow run under a single parent run. Aggregated
    summary metrics are also flattened onto the parent run so the sweep is
    legible in the runs UI without opening the returned object.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness, Safety

        data = [{"inputs": {"question": "What is MLflow?"}}]


        def gpt_4o(question: str) -> str: ...
        def claude(question: str) -> str: ...


        result = mlflow.genai.evaluate_sweep(
            data=data,
            scorers=[Correctness(), Safety()],
            predict_fns={"gpt-4o": gpt_4o, "claude": claude},
            n_repeats=5,
        )
        print(result.comparison_df)
        print("Best on Correctness:", result.best("correctness"))

    Args:
        data: Evaluation dataset, in any format accepted by
            :func:`mlflow.genai.evaluate`. The same data is used for every cell.
        scorers: Scorers to apply. The same scorers are used for every cell.
        predict_fns: The configurations to sweep over. Either a mapping of
            config name to predict function, or a list of predict functions
            (named by ``fn.__name__``, de-duplicated with an index suffix).
            Each function has the same contract as ``evaluate``'s ``predict_fn``.
        n_repeats: Number of times to evaluate each config. Must be >= 1.
            Defaults to 3, the minimum for a meaningful t interval without being
            expensive. With a single repeat, the interval falls back to a
            bootstrap over rows.
        predict_once: When True, run predictions only on the first repeat and
            reuse the resulting traces for the remaining repeats, re-running only
            the scorers. This captures scorer/judge variance but not the model's
            own generation variance, at roughly ``1/n_repeats`` of the prediction
            cost. Requires a predict function per config (traces must be
            generated); has no effect when ``n_repeats == 1``.

    Returns:
        A :class:`~mlflow.genai.evaluation.sweep_entities.SweepResult` with
        per-config confidence intervals, latency percentiles, the underlying
        per-repeat results, and a ``comparison_df``.
    """
    if n_repeats < 1:
        raise MlflowException.invalid_parameter_value(f"n_repeats must be >= 1, got {n_repeats}.")

    named_fns = _normalize_predict_fns(predict_fns)

    parent_run = mlflow.active_run()
    parent_ctx = nullcontext(parent_run) if parent_run else mlflow.start_run()
    with parent_ctx as parent:
        parent_run_id = parent.info.run_id
        if parent.data.tags.get(MLFLOW_RUN_TYPE) is None:
            mlflow.set_tag(MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_GENAI_EVALUATE_SWEEP)

        sweep_result = SweepResult(parent_run_id=parent_run_id, n_repeats=n_repeats)

        for name, predict_fn in named_fns.items():
            config_result = _run_config(
                name=name,
                predict_fn=predict_fn,
                data=data,
                scorers=scorers,
                n_repeats=n_repeats,
                predict_once=predict_once,
            )
            sweep_result.configs[name] = config_result

        _log_summary_metrics_to_parent(sweep_result)

    return sweep_result


def _normalize_predict_fns(
    predict_fns: dict[str, Callable[..., Any]] | list[Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Coerce the predict_fns argument into an ordered {name: fn} mapping."""
    if isinstance(predict_fns, dict):
        if not predict_fns:
            raise MlflowException.invalid_parameter_value("predict_fns must not be empty.")
        return dict(predict_fns)

    if not predict_fns:
        raise MlflowException.invalid_parameter_value("predict_fns must not be empty.")

    named: dict[str, Callable[..., Any]] = {}
    for fn in predict_fns:
        base = getattr(fn, "__name__", None) or "predict_fn"
        name = base
        i = 1
        while name in named:
            name = f"{base}_{i}"
            i += 1
        named[name] = fn
    return named


def _run_config(
    *,
    name: str,
    predict_fn: Callable[..., Any],
    data: "EvaluationDatasetTypes",
    scorers: list[Scorer],
    n_repeats: int,
    predict_once: bool,
) -> SweepConfigResult:
    """Run all repeats for a single config and aggregate them into a result."""
    config_result = SweepConfigResult(name=name)
    reuse_data: Any | None = None  # traces from repeat 0, when predict_once is set

    for repeat in range(n_repeats):
        use_predictions = repeat == 0 or not predict_once
        with mlflow.start_run(nested=True, run_name=f"{name}-repeat-{repeat}") as child:
            if use_predictions:
                result = evaluate(data=data, scorers=scorers, predict_fn=predict_fn)
                if predict_once and n_repeats > 1:
                    reuse_data = _traces_for_reuse(result.run_id)
            else:
                # Re-score the traces generated on repeat 0; no predict_fn.
                result = evaluate(data=reuse_data, scorers=scorers)

            config_result.child_run_ids.append(child.info.run_id)
            config_result.results.append(result)

    _aggregate_config(config_result)
    return config_result


def _traces_for_reuse(run_id: str):
    """Fetch the traces produced by a repeat, as a DataFrame for re-scoring."""
    return mlflow.search_traces(run_id=run_id, return_type="pandas")


def _aggregate_config(config_result: SweepConfigResult) -> None:
    """Populate a config's scorer intervals and latency from its per-repeat results."""
    # Per-repeat mean per scorer, and all row-level values pooled across repeats.
    per_run_means: dict[str, list[float]] = defaultdict(list)
    pooled_row_values: dict[str, list[float]] = defaultdict(list)

    for result in config_result.results:
        df = result.result_df
        if df is None:
            continue
        for column in df.columns:
            if not column.endswith(_VALUE_COLUMN_SUFFIX):
                continue
            scorer_name = column[: -len(_VALUE_COLUMN_SUFFIX)]
            row_values = _numeric_column_values(df[column])
            if not row_values:
                continue
            per_run_means[scorer_name].append(sum(row_values) / len(row_values))
            pooled_row_values[scorer_name].extend(row_values)

    for scorer_name, means in per_run_means.items():
        interval = compute_scorer_interval(
            per_run_means=means,
            row_values=pooled_row_values[scorer_name],
        )
        config_result.scorer_intervals[scorer_name] = ScorerInterval.from_interval(interval)

    config_result.latency = _compute_latency(config_result.child_run_ids)


def _numeric_column_values(series) -> list[float]:
    """Cast a scorer value column to floats, dropping non-numeric / missing entries.

    Booleans and 0/1 numerics pass through so pass/fail scorers are detected
    downstream; pass/fail string ratings ("yes"/"no") are mapped to 1.0/0.0 to
    match the harness's own aggregation casting.
    """
    from mlflow.genai.judges.builtin import CategoricalRating

    values: list[float] = []
    for raw in series.tolist():
        if isinstance(raw, bool):
            values.append(float(raw))
        elif isinstance(raw, (int, float)):
            if raw != raw:  # NaN
                continue
            values.append(float(raw))
        elif isinstance(raw, str):
            rating = CategoricalRating(raw.lower())
            if rating == CategoricalRating.YES:
                values.append(1.0)
            elif rating == CategoricalRating.NO:
                values.append(0.0)
    return values


def _compute_latency(child_run_ids: list[str]) -> LatencyStats | None:
    """Aggregate per-row prediction latency across a config's child runs."""
    durations_ms: list[float] = []
    for run_id in child_run_ids:
        try:
            traces = mlflow.search_traces(run_id=run_id, return_type="list", include_spans=False)
        except Exception as e:
            _logger.debug(f"Failed to fetch traces for latency of run {run_id}: {e}")
            continue
        for trace in traces:
            duration = trace.info.execution_duration
            if duration is not None:
                durations_ms.append(float(duration))

    if not durations_ms:
        return None

    durations_ms.sort()
    return LatencyStats(
        p50=_percentile(durations_ms, 50),
        p90=_percentile(durations_ms, 90),
        p95=_percentile(durations_ms, 95),
        p99=_percentile(durations_ms, 99),
        mean=sum(durations_ms) / len(durations_ms),
        n_rows=len(durations_ms),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _log_summary_metrics_to_parent(sweep_result: SweepResult) -> None:
    """Flatten per-config summary metrics onto the parent run.

    Logs ``{config}/{scorer}/mean|ci_low|ci_high|std`` plus
    ``{config}/latency_p50|p90|p95|p99`` so the sweep is visible in the runs UI.
    Config names are sanitized to the characters MLflow allows in metric keys.
    """
    metrics: dict[str, float] = {}
    for config in sweep_result.configs.values():
        cfg_key = _sanitize_metric_key(config.name)
        for scorer_name, interval in config.scorer_intervals.items():
            scorer_key = _sanitize_metric_key(scorer_name)
            prefix = f"{cfg_key}/{scorer_key}"
            metrics[f"{prefix}/mean"] = interval.mean
            metrics[f"{prefix}/ci_low"] = interval.ci_low
            metrics[f"{prefix}/ci_high"] = interval.ci_high
            metrics[f"{prefix}/std"] = interval.std
        if config.latency is not None:
            metrics[f"{cfg_key}/latency_p50_ms"] = config.latency.p50
            metrics[f"{cfg_key}/latency_p90_ms"] = config.latency.p90
            metrics[f"{cfg_key}/latency_p95_ms"] = config.latency.p95
            metrics[f"{cfg_key}/latency_p99_ms"] = config.latency.p99

    if metrics:
        try:
            mlflow.log_metrics(metrics, run_id=sweep_result.parent_run_id)
        except Exception as e:
            _logger.warning(f"Failed to log sweep summary metrics to parent run: {e}")


def _sanitize_metric_key(key: str) -> str:
    """Replace characters MLflow disallows in metric keys with underscores.

    MLflow permits alphanumerics and ``_-./ space :``. Anything else (e.g. a
    config named ``gpt-4o@v2``) is replaced so metric logging doesn't fail.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ :")
    return "".join(c if c in allowed else "_" for c in key)
