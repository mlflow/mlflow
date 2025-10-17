import functools
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, create_model

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer
from mlflow.tracking.client import MlflowClient

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def prompt_optimization_autolog(
    optimizer_name: str,
    num_prompts: int,
    num_training_samples: int,
    train_data_df: "pd.DataFrame",
):
    """
    Context manager for autologging prompt optimization runs.

    Args:
        optimizer_name: Name of the optimizer being used
        num_prompts: Number of prompts being optimized
        num_training_samples: Number of training samples
        train_data_df: Training data as a pandas DataFrame

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

        # Log training dataset as run input
        dataset = mlflow.data.from_pandas(train_data_df, source="prompt_optimization_train_data")
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


def validate_train_data(train_data: list[dict[str, Any]]) -> None:
    """
    Validate that training data has required fields for prompt optimization.

    Args:
        train_data: List of training data records converted to dict format.

    Raises:
        ValueError: If any record is missing required fields or has empty required fields.
    """
    for i, record in enumerate(train_data):
        if "inputs" not in record or not record["inputs"]:
            raise ValueError(f"Record {i} is missing required 'inputs' field or it is empty")

        # Check that at least one of outputs or expectations is present and not None
        # We explicitly check for None to allow falsy values like False, 0, or empty strings
        has_outputs = record.get("outputs") is not None
        has_expectations = record.get("expectations") is not None

        if not has_outputs and not has_expectations:
            raise ValueError(
                f"Record {i} must have at least one non-empty field: 'outputs' or 'expectations'. "
                f"Found outputs={record.get('outputs')}, expectations={record.get('expectations')}"
            )


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
        element_types = set()
        for item in value:
            element_types.add(infer_type_from_value(item))
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
) -> Callable[[Any, Any, dict[str, Any]], float]:
    """
    Create a metric function from scorers and an optional objective function.

    Args:
        scorers: List of scorers to evaluate inputs, outputs, and expectations.
        objective: Optional function that aggregates scorer outputs into a single score.
                  Takes a dict mapping scorer names to scores and returns a float.
                  If None and all scorers return numerical or CategoricalRating values,
                  uses default aggregation (sum for numerical, conversion for categorical).

    Returns:
        A callable that takes (inputs, outputs, expectations) and returns a float score.

    Raises:
        MlflowException: If scorers return non-numerical values and no objective is provided.
    """
    from mlflow.entities import Feedback
    from mlflow.genai.judges import CategoricalRating

    def _convert_to_numeric(score: Any) -> float | None:
        """Convert a value to numeric, handling CategoricalRating and common types."""
        if isinstance(score, (int, float, bool)):
            return float(score)
        elif isinstance(score, Feedback) and isinstance(score.value, CategoricalRating):
            # Convert CategoricalRating to numeric: YES=1.0, NO=0.0
            return 1.0 if score.value == CategoricalRating.YES else 0.0
        return None

    def metric(
        inputs: Any,
        outputs: Any,
        expectations: dict[str, Any],
    ) -> float:
        scores = {}

        for scorer in scorers:
            scores[scorer.name] = scorer.run(
                inputs=inputs, outputs=outputs, expectations=expectations
            )

        if objective is not None:
            return objective(scores)

        # Try to convert all scores to numeric
        numeric_scores = {}
        for name, score in scores.items():
            numeric_value = _convert_to_numeric(score)
            if numeric_value is not None:
                numeric_scores[name] = numeric_value

        # If all scores were convertible, use sum as default aggregation
        if len(numeric_scores) == len(scores):
            return sum(numeric_scores.values())

        # Otherwise, report error with actual types
        non_convertible = {
            k: type(v).__name__ for k, v in scores.items() if k not in numeric_scores
        }
        scorer_details = ", ".join([f"{k} (type: {t})" for k, t in non_convertible.items()])
        raise MlflowException(
            f"Scorers [{scorer_details}] return non-numerical values that cannot be "
            "automatically aggregated. Please provide an `objective` function to aggregate "
            "these values into a single score for optimization."
        )

    return metric
