"""Entry point to the evaluation harness"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    # If tqdm is not installed, we don't show a progress bar
    tqdm = None

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING,
    MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS,
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
)
from mlflow.genai.evaluation import context
from mlflow.genai.evaluation.entities import EvalItem, EvalResult, EvaluationResult
from mlflow.genai.evaluation.session_utils import (
    classify_scorers,
    evaluate_session_level_scorers,
    group_traces_by_session,
)
from mlflow.genai.evaluation.telemetry import emit_custom_metric_event
from mlflow.genai.evaluation.utils import (
    PGBAR_FORMAT,
    is_none_or_nan,
    make_code_type_assessment_source,
    standardize_scorer_value,
    validate_tags,
)
from mlflow.genai.scorers.aggregation import compute_aggregated_metrics
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.trace_utils import (
    _does_store_support_trace_linking,
    batch_link_traces_to_run,
    clean_up_extra_traces,
    construct_eval_result_df,
    create_minimal_trace,
)
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.tracing.constant import AssessmentMetadataKey, TraceTagKey
from mlflow.tracing.utils.copy import copy_trace_to_experiment
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS

_logger = logging.getLogger(__name__)


def _log_multi_turn_assessments_to_traces(
    multi_turn_assessments: dict[str, list[Feedback]],
    eval_results: list[EvalResult],
    run_id: str,
) -> None:
    """
    Log multi-turn assessments to the first trace of each session.

    Args:
        multi_turn_assessments: Dictionary mapping trace_id to list of assessments
        eval_results: List of EvalResult objects to update with multi-turn assessments
        run_id: MLflow run ID for logging
    """
    for eval_result in eval_results:
        if eval_result.eval_item.trace is None:
            continue

        trace_id = eval_result.eval_item.trace.info.trace_id
        if trace_id not in multi_turn_assessments:
            continue

        assessments_list = multi_turn_assessments[trace_id]
        try:
            _log_assessments(
                run_id=run_id,
                trace=eval_result.eval_item.trace,
                assessments=assessments_list,
            )
            eval_result.assessments.extend(assessments_list)
        except Exception as e:
            _logger.warning(f"Failed to log multi-turn assessments for trace {trace_id}: {e}")


@context.eval_context
def run(
    *,
    eval_df: pd.DataFrame,
    predict_fn=None,
    scorers=None,
    run_id: str | None = None,
) -> EvaluationResult:
    """
    Runs GenAI evaluation harness to the given dataset.

    The overall flow is:

    1. Convert the dataset to a list of EvalItem objects
    2. Classify scorers into single-turn and multi-turn
    3. Run the prediction and single-turn scoring for each EvalItem in parallel
        a. If predict_fn is provided, invoke the predict_fn for the EvalItem
        b. If predict_fn is not provided, create a dummy trace for the EvalItem
        c. Execute the single-turn scorers to compute assessments.
        d. Log the assessments to the trace.
    4. If multi-turn scorers exist, evaluate them on session groups
    5. Compute the aggregated metrics from the assessments.
    """
    eval_items = [EvalItem.from_dataset_row(row) for row in eval_df.to_dict(orient="records")]
    eval_start_time = int(time.time() * 1000)

    run_id = context.get_context().get_mlflow_run_id() if run_id is None else run_id

    # Classify scorers into single-turn and multi-turn
    single_turn_scorers, multi_turn_scorers = classify_scorers(scorers)

    session_groups = group_traces_by_session(eval_items) if multi_turn_scorers else {}

    total_tasks = len(eval_items) + len(session_groups)

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
        thread_name_prefix="MlflowGenAIEvalHarness",
    ) as executor:
        # Submit single-turn tasks
        single_turn_futures = {
            executor.submit(
                _run_single,
                eval_item=eval_item,
                scorers=single_turn_scorers,
                predict_fn=predict_fn,
                run_id=run_id,
            ): i
            for i, eval_item in enumerate(eval_items)
        }

        # Collect results with unified progress bar
        eval_results = [None] * len(eval_items)
        multi_turn_assessments = {}

        # Create progress bar for all tasks
        progress_bar = (
            tqdm(
                total=total_tasks,
                desc="Evaluating",
                smoothing=0,
                bar_format=PGBAR_FORMAT,
            )
            if tqdm is not None
            else None
        )

        try:
            # Phase 1: Complete single-turn tasks
            for future in as_completed(single_turn_futures):
                idx = single_turn_futures[future]
                eval_results[idx] = future.result()
                if progress_bar:
                    progress_bar.update(1)

            # Phase 2: Submit and complete multi-turn tasks (after single-turn)
            # We run multi-turn scorers after single-turn, since single-turn scorers may create new
            # traces that are needed by multi-turn scorers.
            if multi_turn_scorers and session_groups:
                multi_turn_futures = [
                    executor.submit(
                        evaluate_session_level_scorers,
                        session_id=session_id,
                        session_items=session_items,
                        multi_turn_scorers=multi_turn_scorers,
                    )
                    for session_id, session_items in session_groups.items()
                ]

                for future in as_completed(multi_turn_futures):
                    session_result = future.result()
                    multi_turn_assessments.update(session_result)
                    if progress_bar:
                        progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()

    if multi_turn_assessments:
        _log_multi_turn_assessments_to_traces(
            multi_turn_assessments=multi_turn_assessments,
            eval_results=eval_results,
            run_id=run_id,
        )

    # Link traces to the run if the backend support it
    batch_link_traces_to_run(run_id=run_id, eval_results=eval_results)

    # Refresh traces on eval_results to include all logged assessments.
    # This is done once after all assessments (single-turn and multi-turn) are logged to the traces.
    _refresh_eval_result_traces(eval_results)

    # Aggregate metrics and log to MLflow run
    aggregated_metrics = compute_aggregated_metrics(eval_results, scorers=scorers)
    mlflow.log_metrics(aggregated_metrics)

    try:
        emit_custom_metric_event(scorers, len(eval_items), aggregated_metrics)
    except Exception as e:
        _logger.debug(f"Failed to emit custom metric usage event: {e}", exc_info=True)

    # Search for all traces in the run. We need to fetch the traces from backend here to include
    # all traces in the result.
    traces = mlflow.search_traces(run_id=run_id, include_spans=False, return_type="list")

    # Clean up noisy traces generated during evaluation
    clean_up_extra_traces(traces, eval_start_time)

    return EvaluationResult(
        run_id=run_id,
        result_df=construct_eval_result_df(run_id, traces, eval_results),
        metrics=aggregated_metrics,
    )


def _run_single(
    eval_item: EvalItem,
    scorers: list[Scorer],
    run_id: str | None,
    predict_fn: Callable[..., Any] | None = None,
) -> EvalResult:
    """Run the logic of the eval harness for a single eval item."""
    # Set the MLflow run ID in the context for this thread
    if run_id:
        # Manually set the mlflow_run_id for this context to be the same as was set in
        # the parent thread. This is required because MLflow runs are thread-local.
        ctx = context.get_context()
        ctx.set_mlflow_run_id(run_id)

    if predict_fn:
        # NB: Setting prediction context let us retrieve the trace by a custom ID. Setting
        # is_evaluate=True disables async trace logging to make sure the trace is available.
        eval_request_id = str(uuid.uuid4())
        with set_prediction_context(Context(request_id=eval_request_id, is_evaluate=True)):
            try:
                eval_item.outputs = predict_fn(eval_item.inputs)
            except Exception as e:
                eval_item.error_message = (
                    f"Failed to invoke the predict_fn with {eval_item.inputs}: {e}"
                )

        eval_item.trace = mlflow.get_trace(eval_request_id, silent=True)
    elif eval_item.trace is not None:
        if _should_clone_trace(eval_item.trace, run_id):
            try:
                trace_id = copy_trace_to_experiment(eval_item.trace.to_dict())
                eval_item.trace = mlflow.get_trace(trace_id)
            except Exception as e:
                eval_item.error_message = f"Failed to clone trace to the current experiment: {e}"
        else:
            MlflowClient().link_traces_to_run([eval_item.trace.info.trace_id], run_id=run_id)
    else:
        # When static dataset (a pair of inputs and outputs) is given, we create a minimal
        # trace with root span only, to log the assessments on it.
        minimal_trace = create_minimal_trace(eval_item)
        eval_item.trace = minimal_trace

    # Execute the scorers
    assessments = _compute_eval_scores(eval_item=eval_item, scorers=scorers)
    assessments.extend(_get_new_expectations(eval_item))
    eval_result = EvalResult(eval_item=eval_item, assessments=assessments)

    tags = eval_item.tags if not is_none_or_nan(eval_item.tags) else {}
    validate_tags(tags)

    for key in tags.keys() - IMMUTABLE_TAGS:
        try:
            mlflow.set_trace_tag(trace_id=eval_item.trace.info.trace_id, key=key, value=tags[key])
        except Exception as e:
            # Failures in logging to MLflow should not fail the entire harness run
            _logger.warning(f"Failed to log tag {key} to MLflow: {e}")

    try:
        _log_assessments(
            run_id=run_id,
            trace=eval_item.trace,
            assessments=eval_result.assessments,
        )
    except Exception as e:
        # Failures in logging to MLflow should not fail the entire harness run
        _logger.warning(f"Failed to log trace and assessments to MLflow: {e}")

    return eval_result


def _compute_eval_scores(
    *,
    eval_item: EvalItem,
    scorers: list[Scorer],
) -> list[Feedback]:
    """Compute the per-eval-item scores."""
    if not scorers:
        return []

    def run_scorer(scorer):
        try:
            scorer_func = scorer.run

            if MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING.get():
                scorer_func = mlflow.trace(name=scorer.name, span_type=SpanType.EVALUATOR)(
                    scorer_func
                )

            value = scorer_func(
                inputs=eval_item.inputs,
                outputs=eval_item.outputs,
                expectations=eval_item.expectations,
                trace=eval_item.trace,
            )
            feedbacks = standardize_scorer_value(scorer.name, value)

        except Exception as e:
            feedbacks = [
                Feedback(
                    name=scorer.name,
                    source=make_code_type_assessment_source(scorer.name),
                    error=AssessmentError(
                        error_code="SCORER_ERROR",
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                    ),
                )
            ]

        # Record the trace ID for the scorer function call.
        if MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING.get() and (
            trace_id := mlflow.get_last_active_trace_id(thread_local=True)
        ):
            for feedback in feedbacks:
                feedback.metadata = {
                    **(feedback.metadata or {}),
                    AssessmentMetadataKey.SCORER_TRACE_ID: trace_id,
                }
            # Set the scorer name tag to the trace to identify the trace is generated by a scorer.
            mlflow.set_trace_tag(
                trace_id=trace_id,
                key=TraceTagKey.SOURCE_SCORER_NAME,
                value=scorer.name,
            )
        return feedbacks

    # Use a thread pool to run scorers in parallel
    # Limit concurrent scorers to prevent rate limiting errors with external LLM APIs
    max_scorer_workers = min(len(scorers), MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS.get())
    with ThreadPoolExecutor(
        max_workers=max_scorer_workers,
        thread_name_prefix="MlflowGenAIEvalScorer",
    ) as executor:
        futures = [executor.submit(run_scorer, scorer) for scorer in scorers]

        try:
            results = [future.result() for future in as_completed(futures)]
        except KeyboardInterrupt:
            # Cancel pending futures
            executor.shutdown(cancel_futures=True)
            raise

    # Flatten list[list[Assessment]] into a single list[Assessment]
    return [assessment for sublist in results for assessment in sublist]


def _get_new_expectations(eval_item: EvalItem) -> list[Expectation]:
    existing_expectations = {
        a.name for a in eval_item.trace.info.assessments if a.expectation is not None
    }
    return [
        exp
        for exp in eval_item.get_expectation_assessments()
        if exp.name not in existing_expectations
    ]


def _log_assessments(
    run_id: str | None,
    trace: Trace,
    assessments: list[Assessment],
) -> None:
    """
    Log assessments to a trace.
    """
    for assessment in assessments:
        # Ensure that if we created a new trace, that the updated trace_id is reflected in
        # the assessments.
        assessment.trace_id = trace.info.trace_id
        if run_id is not None:
            assessment.metadata = {
                **(assessment.metadata or {}),
                AssessmentMetadataKey.SOURCE_RUN_ID: run_id,
            }

        # NB: Root span ID is necessarily to show assessment results in DBX eval UI.
        if root_span := trace.data._get_root_span():
            assessment.span_id = root_span.span_id
        else:
            _logger.debug(f"No root span found for trace {trace.info.trace_id}")

        mlflow.log_assessment(trace_id=assessment.trace_id, assessment=assessment)


def _refresh_eval_result_traces(eval_results: list[EvalResult]) -> None:
    """
    Refresh traces on eval_results to include logged assessments.

    This function fetches the updated traces from the backend after all assessments
    (both single-turn and multi-turn) have been logged.
    """

    def _fetch_trace(eval_result: EvalResult):
        if eval_result.eval_item.trace is None:
            return None
        trace_id = eval_result.eval_item.trace.info.trace_id
        try:
            return eval_result, mlflow.get_trace(trace_id)
        except Exception as e:
            _logger.warning(f"Failed to refresh trace {trace_id}: {e}")
            return None

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
        thread_name_prefix="GenAIEvaluationTraceRefresh",
    ) as executor:
        futures = [executor.submit(_fetch_trace, er) for er in eval_results]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                eval_result, refreshed_trace = result
                eval_result.eval_item.trace = refreshed_trace


def _should_clone_trace(trace: Trace | None, run_id: str | None) -> bool:
    from mlflow.tracking.fluent import _get_experiment_id

    if trace is None:
        return False

    # If the trace is stored in UC table, we don't clone the trace
    if trace.info.trace_location.uc_schema is not None:
        return False

    # Check if the trace is from the same experiment. If it isn't, we need to clone the trace
    trace_experiment = trace.info.trace_location.mlflow_experiment
    current_experiment = _get_experiment_id()
    if trace_experiment is not None and trace_experiment.experiment_id != current_experiment:
        return True

    # If the backend doesn't support trace<->run linking, need to clone the trace to the new run.
    return not _does_store_support_trace_linking(
        tracking_uri=mlflow.get_tracking_uri(),
        trace=trace,
        run_id=run_id,
    )
