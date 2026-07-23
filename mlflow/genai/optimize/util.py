from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, create_model

from mlflow.entities import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers
from mlflow.tracking.client import MlflowClient

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def prompt_optimization_autolog(
    optimizer_name: str,
    num_prompts: int,
    num_training_samples: int,
    train_data_df: "pd.DataFrame" | None,
):
    """
    Context manager for autologging prompt optimization runs.

    Args:
        optimizer_name: Name of the optimizer being used
        num_prompts: Number of prompts being optimized
        num_training_samples: Number of training samples
        train_data_df: Training data as a pandas DataFrame. If None or empty, it means zero-shot
            optimization.

    Yields:
        Tuple of (run_id, results_dict) where results_dict should be populated with
        PromptOptimizerOutput and list of optimized PromptVersion objects
    """
    import mlflow.data

    active_run = mlflow.active_run()
    run_context = mlflow.start_run() if active_run is None else nullcontext(active_run)

    with run_context as run:
        client = MlflowClient()
        run_id = run.info.run_id

        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("num_prompts", num_prompts)
        mlflow.log_param("num_training_samples", num_training_samples)

        if train_data_df is not None and not train_data_df.empty:
            # Log training dataset as run input if it is provided
            dataset = mlflow.data.from_pandas(
                train_data_df, source="prompt_optimization_train_data"
            )
            mlflow.log_input(dataset, context="training")

        results = {}
        yield results

        if "optimized_prompts" in results:
            for prompt in results["optimized_prompts"]:
                client.link_prompt_version_to_run(run_id=run_id, prompt=prompt)

        if "optimizer_output" in results:
            output = results["optimizer_output"]
            if output.initial_eval_score is not None:
                mlflow.log_metric("initial_eval_score", output.initial_eval_score)
            if output.final_eval_score is not None:
                mlflow.log_metric("final_eval_score", output.final_eval_score)
            if output.initial_eval_score_per_scorer:
                mlflow.log_metrics({
                    f"initial_eval_score.{scorer_name}": score
                    for scorer_name, score in output.initial_eval_score_per_scorer.items()
                })
            if output.final_eval_score_per_scorer:
                mlflow.log_metrics({
                    f"final_eval_score.{scorer_name}": score
                    for scorer_name, score in output.final_eval_score_per_scorer.items()
                })


def validate_train_data(
    train_data: "pd.DataFrame",
    scorers: list[Scorer] | None,
    predict_fn: Callable[..., Any] | None = None,
) -> None:
    """
    Validate that training data has required fields for prompt optimization.

    Args:
        train_data: Training data as a pandas DataFrame.
        scorers: Scorers to validate the training data for. Can be None for zero-shot mode.
        predict_fn: The predict function to validate the training data for.

    Raises:
        MlflowException: If any record is missing required 'inputs' field or it is empty.
    """
    for i, record in enumerate(train_data.to_dict("records")):
        if "inputs" not in record or not record["inputs"]:
            raise MlflowException.invalid_parameter_value(
                f"Record {i} is missing required 'inputs' field or it is empty"
            )

    if scorers is not None:
        builtin_scorers = [scorer for scorer in scorers if isinstance(scorer, BuiltInScorer)]
        valid_data_for_builtin_scorers(train_data, builtin_scorers, predict_fn)


def infer_type_from_value(value: Any, model_name: str = "Output") -> type:
    """
    Infer the type from the value.
    Only supports primitive types, lists, and dict and Pydantic models.
    """
    if value is None:
        return type(None)
    elif isinstance(value, (bool, int, float, str)):
        return type(value)
    elif isinstance(value, list):
        if not value:
            return list[Any]
        element_types = {infer_type_from_value(item) for item in value}
        return list[functools.reduce(lambda x, y: x | y, element_types)]
    elif isinstance(value, dict):
        fields = {k: (infer_type_from_value(v, model_name=k), ...) for k, v in value.items()}
        return create_model(model_name, **fields)
    elif isinstance(value, BaseModel):
        return type(value)
    return Any


def create_metric_from_scorers(
    scorers: list[Scorer],
    objective: Callable[[dict[str, Any]], float] | None = None,
) -> Callable[[Any, Any, dict[str, Any]], tuple[float, dict[str, str], dict[str, float]]]:
    """
    Create a metric function from scorers and an optional objective function.

    This function creates a metric that mirrors evaluate()'s scorer execution
    pattern (harness.py:839-864): per-scorer exception handling, assessment
    logging on traces, and tracing disabled during scorer execution so judge
    LLM calls don't create stray top-level traces.

    Args:
        scorers: List of scorers to evaluate inputs, outputs, and expectations.
        objective: Optional function that aggregates scorer outputs into a single score.
                  Takes a dict mapping scorer names to scores and returns a float.
                  If None and all scorers return numerical or CategoricalRating values,
                  uses default aggregation (sum for numerical, conversion for categorical).

    Returns:
        A callable that takes (inputs, outputs, expectations, trace) and
        returns a tuple of (aggregated_score, rationales, individual_scores).

    Raises:
        MlflowException: If scorers return non-numerical values and no objective is provided.
    """
    import logging
    import traceback

    import mlflow
    from mlflow.entities import Feedback, SpanType
    from mlflow.entities.assessment_error import AssessmentError
    from mlflow.environment_variables import MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING
    from mlflow.genai.evaluation.harness import _log_assessments
    from mlflow.genai.evaluation.utils import (
        make_code_type_assessment_source,
        standardize_scorer_value,
    )
    from mlflow.genai.judges import CategoricalRating
    from mlflow.tracing.constant import TraceTagKey

    _logger = logging.getLogger(__name__)
    should_trace = MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING.get()

    def _convert_to_numeric(score: Any) -> float | None:
        """Convert a value to numeric, handling Feedback and primitive types."""
        if isinstance(score, Feedback):
            score = score.value
        if score == CategoricalRating.YES:
            return 1.0
        elif score == CategoricalRating.NO:
            return 0.0
        elif isinstance(score, (int, float, bool)):
            return float(score)
        return None

    def metric(
        inputs: Any,
        outputs: Any,
        expectations: dict[str, Any],
        trace: Trace | None,
    ) -> tuple[float, dict[str, str], dict[str, float]]:
        all_feedbacks = []
        scores = {}
        rationales = {}

        try:
            for scorer in scorers:
                # Per-scorer try/except matching evaluate()'s pattern
                # in harness.py:839-864.
                try:
                    scorer_func = scorer.run

                    # Conditionally wrap scorer with tracing, matching
                    # evaluate()'s pattern in harness.py:843-846.
                    if should_trace:
                        scorer_func = mlflow.trace(name=scorer.name, span_type=SpanType.EVALUATOR)(
                            scorer_func
                        )

                    value = scorer_func(
                        inputs=inputs,
                        outputs=outputs,
                        expectations=expectations,
                        trace=trace,
                    )
                    feedbacks = standardize_scorer_value(scorer.name, value)
                except Exception as e:
                    _logger.warning(
                        f"Scorer '{scorer.name}' failed during optimization: "
                        f"{type(e).__name__}: {e}",
                        exc_info=_logger.isEnabledFor(logging.DEBUG),
                    )
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

                # Record scorer trace ID on feedbacks, matching
                # harness.py:867-878.
                if should_trace and (
                    trace_id := mlflow.get_last_active_trace_id(thread_local=True)
                ):
                    for feedback in feedbacks:
                        feedback.metadata = {
                            **(feedback.metadata or {}),
                            "scorer_trace_id": trace_id,
                        }
                    mlflow.set_trace_tag(
                        trace_id=trace_id,
                        key=TraceTagKey.SOURCE_SCORER_NAME,
                        value=scorer.name,
                    )

                all_feedbacks.extend(feedbacks)

                for fb in feedbacks:
                    name = fb.name or scorer.name
                    scores[name] = fb
                    if fb.rationale:
                        rationales[name] = fb.rationale

            # Log assessments on the trace, matching evaluate()'s behavior.
            if trace is not None:
                try:
                    active_run = mlflow.active_run()
                    run_id = active_run.info.run_id if active_run else None
                    _log_assessments(run_id=run_id, trace=trace, assessments=all_feedbacks)
                except Exception as e:
                    _logger.debug(f"Failed to log assessments: {e}")

            # Try to convert all scores to numeric
            numeric_scores = {}
            for name, score in scores.items():
                numeric_value = _convert_to_numeric(score)
                if numeric_value is not None:
                    numeric_scores[name] = numeric_value

            if objective is not None:
                return objective(scores), rationales, numeric_scores

            # If all scores were convertible, use average as default aggregation
            if len(numeric_scores) == len(scores):
                aggregated = sum(numeric_scores.values()) / len(numeric_scores)
                return aggregated, rationales, numeric_scores

            # Non-convertible scores without objective -- return 0
            return 0.0, rationales, numeric_scores

        except Exception as e:
            # Catch-all for unexpected errors in the scoring pipeline
            _logger.warning(
                f"Scoring pipeline failed: {type(e).__name__}: {e}. Returning score=0.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            scorer_names = [s.name for s in scorers]
            zero_scores = dict.fromkeys(scorer_names, 0.0)
            error_rationales = {name: f"Error: {type(e).__name__}: {e}" for name in scorer_names}
            return 0.0, error_rationales, zero_scores

    return metric
