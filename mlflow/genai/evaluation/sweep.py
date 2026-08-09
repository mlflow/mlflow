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
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable
from unittest import mock

import mlflow
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.base import evaluate
from mlflow.genai.evaluation.statistics import compute_scorer_interval
from mlflow.genai.evaluation.sweep_entities import (
    CostStats,
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
    with parent_ctx as parent, _sweep_execution_context():
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


@contextmanager
def _sweep_execution_context():
    """Configure the process for the duration of a sweep.

    Suppresses the per-cell "view evaluation results" output — a sweep runs many
    nested ``evaluate`` calls, and rendering that summary (and its REST fetch)
    for every cell floods a notebook and can hang it. Also defaults
    ``MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION`` on so a sweep doesn't invoke each
    predict_fn on the first dataset sample before the run starts; an explicit
    user setting is left untouched.
    """
    skip_validation = MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.is_set()
    if not skip_validation:
        MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.set(True)
    try:
        with mock.patch(
            "mlflow.genai.evaluation.base.display_evaluation_output", lambda *a, **k: None
        ):
            yield
    finally:
        if not skip_validation:
            MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.unset()


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

    # The sweep defaults MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION on, which also skips
    # evaluate's auto-tracing of an untraced predict_fn. Wrap it here so untraced
    # functions still emit the trace scorers need, without the pre-run sample call.
    if not getattr(predict_fn, "__mlflow_traced__", False):
        predict_fn = mlflow.trace(predict_fn)

    for repeat in range(n_repeats):
        use_predictions = repeat == 0 or not predict_once
        # A single cell failing (bad endpoint, transient error) must not abort the
        # whole sweep — record it and move on to the next repeat/config.
        try:
            with mlflow.start_run(nested=True, run_name=f"{name}-repeat-{repeat}") as child:
                if use_predictions:
                    result = evaluate(data=data, scorers=scorers, predict_fn=predict_fn)
                    if predict_once and n_repeats > 1:
                        reuse_data = _traces_for_reuse(result.run_id)
                elif reuse_data is None:
                    # predict_once, but repeat 0 failed to produce reusable traces.
                    raise MlflowException(
                        f"Cannot reuse predictions for '{name}' repeat {repeat}: "
                        "repeat 0 did not produce traces."
                    )
                else:
                    result = evaluate(data=reuse_data, scorers=scorers)

                config_result.child_run_ids.append(child.info.run_id)
                config_result.results.append(result)
        except Exception as e:
            config_result.failed_repeats += 1
            _logger.warning(f"Config '{name}' repeat {repeat} failed, skipping it: {e}")

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

    config_result.latency, config_result.cost = _compute_trace_stats(config_result.child_run_ids)


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


def _compute_trace_stats(
    child_run_ids: list[str],
) -> tuple[LatencyStats | None, CostStats | None]:
    """Aggregate per-request latency and cost across a config's child-run traces.

    Each trace is one request. Latency comes from the trace's execution duration
    and cost from its ``cost`` metadata (USD, from token usage x LiteLLM
    pricing). Cost is per request, not per token, and is ``None`` when no trace
    carries cost data (e.g. the provider doesn't report token usage).
    """
    durations_ms: list[float] = []
    costs_usd: list[float] = []
    for run_id in child_run_ids:
        try:
            traces = mlflow.search_traces(run_id=run_id, return_type="list", include_spans=False)
        except Exception as e:
            _logger.debug(f"Failed to fetch traces for run {run_id}: {e}")
            continue
        for trace in traces:
            if (duration := trace.info.execution_duration) is not None:
                durations_ms.append(float(duration))
            if (cost := trace.info.cost) and (total := cost.get("total_cost")) is not None:
                costs_usd.append(float(total))

    latency = None
    if durations_ms:
        durations_ms.sort()
        latency = LatencyStats(
            p50=_percentile(durations_ms, 50),
            p90=_percentile(durations_ms, 90),
            p95=_percentile(durations_ms, 95),
            p99=_percentile(durations_ms, 99),
            mean=sum(durations_ms) / len(durations_ms),
            n_rows=len(durations_ms),
        )

    cost = None
    if costs_usd:
        cost = CostStats(
            mean_per_request=sum(costs_usd) / len(costs_usd),
            total=sum(costs_usd),
            n_rows=len(costs_usd),
        )

    return latency, cost


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

    Logs ``{config}/{scorer}/mean|ci_low|ci_high|std``, ``{config}/latency_pXX``,
    and ``{config}/cost_per_request_usd`` so the sweep is visible in the runs UI.
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
        if config.cost is not None:
            metrics[f"{cfg_key}/cost_per_request_usd"] = config.cost.mean_per_request

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
