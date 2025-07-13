"""Entry point to the evaluation harness"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import pandas as pd

import mlflow
from mlflow.entities.assessment import Assessment, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.trace import Trace
from mlflow.genai.evaluation import context
from mlflow.genai.evaluation.entities import EvalItem, EvalResult, EvaluationResult
from mlflow.genai.evaluation.utils import make_code_type_assessment_source, standardize_scorer_value
from mlflow.genai.scorers.aggregation import compute_aggregated_metrics
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.trace_utils import create_minimal_trace
from mlflow.tracing.constant import AssessmentMetadataKey

_logger = logging.getLogger(__name__)


@context.eval_context
def run(
    *,
    dataset: pd.DataFrame,
    predict_fn=None,
    scorers=None,
) -> EvaluationResult:
    """
    Runs GenAI evaluation harness to the given dataset.

    The overall flow is:

    1. Convert the dataset to a list of EvalItem objects
    2. Run the prediction and scoring for each EvalItem in parallel
        a. If predict_fn is provided, invoke the predict_fn for the EvalItem
        b. If predict_fn is not provided, create a dummy trace for the EvalItem
        c. Execute the scorers to compute assessments.
        d. Log the assessments to the trace.
    3. Compute the aggregated metrics from the assessments.
    """
    eval_items = [EvalItem.from_dataset_row(row) for row in dataset.to_dict(orient="records")]

    ctx = context.get_context()
    run_id = ctx.get_mlflow_run_id()

    with ThreadPoolExecutor(
        # TODO: Add new MLflow environment variable for this
        max_workers=os.environ.get("RAG_EVAL_MAX_WORKERS", 10),
        thread_name_prefix="MlflowGenAIEvalHarness",
    ) as executor:
        futures = [
            executor.submit(
                _run_single,
                eval_item=eval_item,
                scorers=scorers,
                predict_fn=predict_fn,
                run_id=run_id,
            )
            for eval_item in eval_items
        ]
        # TODO: Port the fancy tqdm progress bar from the DBX agent harness.
        eval_results = [future.result() for future in as_completed(futures)]

    # Aggregate metrics and log to MLflow run
    aggregated_metrics = compute_aggregated_metrics(eval_results, scorers=scorers)
    mlflow.log_metrics(aggregated_metrics)

    eval_results_df = pd.DataFrame([result.to_pd_series() for result in eval_results])
    return EvaluationResult(
        run_id=run_id,
        result_df=eval_results_df,
        metrics=aggregated_metrics,
    )


def _run_single(
    eval_item: EvalItem,
    scorers: list[Scorer],
    run_id: Optional[str],
    predict_fn: Optional[Callable[..., Any]] = None,
) -> EvalResult:
    """Run the logic of the eval harness for a single eval item."""
    # Set the MLflow run ID in the context for this thread
    if run_id:
        # Manually set the mlflow_run_id for this context to be the same as was set in
        # the parent thread. This is required because MLflow runs are thread-local.
        ctx = context.get_context()
        ctx.set_mlflow_run_id(run_id)

    # When static dataset (a pair of inputs and outputs) is given, we create a minimal trace
    # with root span only, to log the assessments on it.
    #
    # TODO: Support two more patterns that are currently supported in the DBX agent harness.
    #  1. predict_fn is given
    #  2. traces are given as dataset
    minimal_trace = create_minimal_trace(eval_item)
    eval_item.trace = minimal_trace

    # Execute the scorers
    assessments = _compute_eval_scores(eval_item=eval_item, scorers=scorers)
    assessments.extend(eval_item.get_expectation_assessments())
    eval_result = EvalResult(eval_item=eval_item, assessments=assessments)

    try:
        logged_trace = _log_assessments(
            run_id=run_id,
            trace=eval_item.trace,
            assessments=eval_result.assessments,
        )
        eval_result.eval_item.trace = logged_trace
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
            value = scorer.run(
                inputs=eval_item.inputs,
                outputs=eval_item.outputs,
                expectations=eval_item.expectations,
                trace=eval_item.trace,
            )
            return standardize_scorer_value(scorer.name, value)
        except Exception as e:
            error_assessment = Feedback(
                name=scorer.name,
                source=make_code_type_assessment_source(scorer.name),
                error=AssessmentError(
                    error_code="SCORER_ERROR",
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                ),
            )
            return [error_assessment]

    # Use a thread pool to run scorers in parallel
    with ThreadPoolExecutor(
        max_workers=len(scorers),
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


def _log_assessments(
    run_id: Optional[str],
    trace: Trace,
    assessments: list[Assessment],
) -> Trace:
    for assessment in assessments:
        # Ensure that if we created a new trace, that the updated trace_id is reflected in
        # the assessments.
        assessment.trace_id = trace.info.trace_id
        if run_id is not None:
            assessment.metadata = {
                **(assessment.metadata or {}),
                AssessmentMetadataKey.SOURCE_RUN_ID: run_id,
            }
        mlflow.log_assessment(trace_id=assessment.trace_id, assessment=assessment)

    # Get the trace to fetch newly created assessments.
    return mlflow.get_trace(trace.info.trace_id)
