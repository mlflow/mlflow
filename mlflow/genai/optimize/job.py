"""
Job function for async prompt optimization.

This module provides the job function for running prompt optimization asynchronously
via the MLflow server job execution framework.
"""

import json
import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers import BasePromptOptimizer, GepaPromptOptimizer
from mlflow.genai.scorers.base import Scorer
from mlflow.server.jobs import job

_logger = logging.getLogger(__name__)

_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS = 2
SUPPORTED_OPTIMIZER_TYPES = {"gepa"}


def _create_optimizer(
    optimizer_type: str,
    optimizer_config_json: str | None,
) -> BasePromptOptimizer:
    """
    Create an optimizer instance from type string and config JSON.

    Args:
        optimizer_type: The optimizer type string (e.g., "gepa").
        optimizer_config_json: JSON string of optimizer-specific configuration.

    Returns:
        An instantiated optimizer.

    Raises:
        MlflowException: If optimizer type is not supported.
    """
    config = json.loads(optimizer_config_json) if optimizer_config_json else {}
    optimizer_type_lower = optimizer_type.lower() if optimizer_type else ""

    if optimizer_type_lower == "gepa":
        reflection_model = config.get("reflection_model")
        if not reflection_model:
            raise MlflowException.invalid_parameter_value(
                "Missing required optimizer configuration: 'reflection_model' must be specified "
                "in optimizer_config_json for the GEPA optimizer (e.g., 'openai:/gpt-4o')."
            )
        return GepaPromptOptimizer(
            reflection_model=reflection_model,
            max_metric_calls=config.get("max_metric_calls", 100),
            display_progress_bar=config.get("display_progress_bar", False),
            gepa_kwargs=config.get("gepa_kwargs"),
        )
    elif not optimizer_type_lower:
        raise MlflowException.invalid_parameter_value(
            f"Optimizer type must be specified. Supported types: {SUPPORTED_OPTIMIZER_TYPES}"
        )
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unsupported optimizer type: '{optimizer_type}'. "
            f"Supported types: {SUPPORTED_OPTIMIZER_TYPES}"
        )


def _load_scorers(scorer_names: list[str], experiment_id: str) -> list[Scorer]:
    """
    Load scorers by name.

    For each scorer name, first tries to load it as a built-in scorer (by class name),
    and if not found, falls back to loading from the registered scorer store.

    Args:
        scorer_names: List of scorer names. Can be built-in scorer class names
            (e.g., "Correctness", "Safety") or registered scorer names.
        experiment_id: The experiment ID to load registered scorers from.

    Returns:
        List of Scorer instances.

    Raises:
        MlflowException: If a scorer cannot be found as either built-in or registered.
    """
    from mlflow.genai.scorers import builtin_scorers
    from mlflow.genai.scorers.registry import get_scorer

    scorers = []
    for name in scorer_names:
        # First, try to load as a built-in scorer by class name
        scorer_class = getattr(builtin_scorers, name, None)
        if scorer_class is not None:
            try:
                scorer = scorer_class()
                scorers.append(scorer)
                _logger.info(f"Loaded built-in scorer: {name}")
                continue
            except (TypeError, Exception) as e:
                _logger.debug(f"Failed to instantiate built-in scorer {name}: {e}")

        # Fall back to loading from the registered scorer store
        try:
            scorer = get_scorer(name=name, experiment_id=experiment_id)
            scorers.append(scorer)
            _logger.info(f"Loaded registered scorer: {name}")
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Scorer '{name}' not found. It is neither a built-in scorer "
                f"(e.g., 'Correctness', 'Safety') nor a registered scorer in "
                f"experiment '{experiment_id}'. Error: {e}"
            )

    return scorers


def _build_predict_fn(prompt_uri: str):
    """
    Build a predict function for single-prompt optimization.

    This creates a simple LLM call using the prompt's model configuration.
    The predict function loads the prompt, formats it with inputs, and
    calls the LLM via litellm.

    Args:
        prompt_uri: The URI of the prompt to use for prediction.

    Returns:
        A callable that takes inputs dict and returns the LLM response.
    """
    import litellm

    from mlflow.genai.prompts import load_prompt

    prompt = load_prompt(prompt_uri)
    try:
        model_config = prompt.model_config
        provider = model_config["provider"]
        model_name = model_config["model_name"]
    except Exception:
        raise MlflowException(
            f"Prompt {prompt_uri} doesn't have a model configuration that sets provider and "
            "model_name, which are required for optimization."
        )

    litellm_model = f"{provider}/{model_name}"

    def predict_fn(**kwargs: Any) -> Any:
        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt.format(**kwargs)}],
        )
        return response.choices[0].message.content

    return predict_fn


def _load_train_data_from_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """
    Load training data from an MLflow EvaluationDataset.

    Args:
        dataset_id: The ID of the EvaluationDataset to load.

    Returns:
        List of dicts with 'inputs', 'outputs', and 'expectations' fields.

    Raises:
        MlflowException: If the dataset cannot be loaded or has no records.
    """
    from mlflow.genai.datasets import get_dataset

    dataset = get_dataset(dataset_id=dataset_id)
    df = dataset.to_df()

    if df.empty:
        raise MlflowException.invalid_parameter_value(
            f"Dataset {dataset_id} has no records. Please add records before optimization."
        )

    # Convert DataFrame to list of dicts, keeping only relevant columns
    records = []
    for _, row in df.iterrows():
        record = {"inputs": row["inputs"]}
        if "outputs" in row and row["outputs"] is not None:
            record["outputs"] = row["outputs"]
        if "expectations" in row and row["expectations"] is not None:
            record["expectations"] = row["expectations"]
        records.append(record)

    return records


@job(name="optimize_prompts", max_workers=_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS)
def optimize_prompts_job(
    experiment_id: str,
    prompt_uri: str,
    dataset_id: str,
    optimizer_type: str,
    optimizer_config_json: str | None,
    scorers: list[str],
) -> dict[str, Any]:
    """
    Job function for async single-prompt optimization.

    This function is executed as a background job by the MLflow server.
    It builds a predict_fn from the prompt's model configuration and calls
    mlflow.genai.optimize_prompts() with the provided configuration.

    Note: This job only supports single-prompt optimization. The predict_fn
    is automatically built using the prompt's model_config (provider/model_name)
    via litellm, making the optimization self-contained without requiring users
    to serialize their own predict function.

    Args:
        experiment_id: The experiment ID to track the optimization in.
        prompt_uri: The URI of the prompt to optimize.
        dataset_id: The ID of the EvaluationDataset containing training data.
        optimizer_type: The optimizer type string (e.g., "gepa").
        optimizer_config_json: JSON string of optimizer-specific configuration.
        scorers: List of scorer names. Can be built-in scorer class names
            (e.g., "Correctness", "Safety") or registered scorer names from
            the experiment's scorer registry.

    Returns:
        Dict containing optimization results:
        - optimized_prompt_uri: URI of the optimized prompt
        - optimizer_name: Name of the optimizer used
        - initial_eval_score: Initial evaluation score (if available)
        - final_eval_score: Final evaluation score (if available)
    """
    import mlflow
    from mlflow.genai.optimize import optimize_prompts

    # Set the experiment for tracking
    mlflow.set_experiment(experiment_id=experiment_id)

    # Load training data from the EvaluationDataset
    train_data = _load_train_data_from_dataset(dataset_id)
    _logger.info(f"Loaded {len(train_data)} training samples from dataset {dataset_id}")

    # Build predict_fn from the prompt's model config
    predict_fn = _build_predict_fn(prompt_uri)
    _logger.info(f"Built predict_fn from prompt {prompt_uri}")

    # Create optimizer
    optimizer = _create_optimizer(optimizer_type, optimizer_config_json)
    _logger.info(f"Created optimizer {optimizer_type}")

    # Load scorers by name (built-in or registered)
    loaded_scorers = _load_scorers(scorers, experiment_id)
    _logger.info(f"Loaded {len(loaded_scorers)} scorers: {scorers}")

    # Run optimization (single prompt)
    result = optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_data,
        prompt_uris=[prompt_uri],
        optimizer=optimizer,
        scorers=loaded_scorers,
        enable_tracking=True,
    )

    return {
        "optimized_prompt_uri": result.optimized_prompts[0].uri
        if result.optimized_prompts
        else None,
        "optimizer_name": result.optimizer_name,
        "initial_eval_score": result.initial_eval_score,
        "final_eval_score": result.final_eval_score,
        "dataset_id": dataset_id,
        "scorers": scorers,
    }
