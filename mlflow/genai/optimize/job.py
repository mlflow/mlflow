import json
import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import get_dataset
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.optimize.optimizers import (
    BasePromptOptimizer,
    GepaPromptOptimizer,
    MetaPromptOptimizer,
)
from mlflow.genai.prompts import load_prompt
from mlflow.genai.scorers import builtin_scorers
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.registry import get_scorer
from mlflow.server.jobs import job
from mlflow.telemetry.events import OptimizePromptsJobEvent
from mlflow.telemetry.track import _record_event
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import set_experiment, start_run

_logger = logging.getLogger(__name__)

_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS = 2
SUPPORTED_OPTIMIZER_TYPES = {"gepa", "metaprompt"}


def _create_optimizer(
    optimizer_type: str,
    optimizer_config_json: str | None,
) -> BasePromptOptimizer:
    """
    Create an optimizer instance from type string and config JSON.

    Args:
        optimizer_type: The optimizer type string (e.g., "gepa", "metaprompt").
        optimizer_config_json: JSON string of optimizer-specific configuration.

    Returns:
        An instantiated optimizer.

    Raises:
        MlflowException: If optimizer type is not supported.
    """
    config = json.loads(optimizer_config_json) if optimizer_config_json else {}
    optimizer_type = optimizer_type.lower() if optimizer_type else ""

    if optimizer_type == "gepa":
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
    elif optimizer_type == "metaprompt":
        reflection_model = config.get("reflection_model")
        if not reflection_model:
            raise MlflowException.invalid_parameter_value(
                "Missing required optimizer configuration: 'reflection_model' must be specified "
                "in optimizer_config_json for the MetaPrompt optimizer (e.g., 'openai:/gpt-4o')."
            )
        return MetaPromptOptimizer(
            reflection_model=reflection_model,
            lm_kwargs=config.get("lm_kwargs"),
            guidelines=config.get("guidelines"),
        )
    elif not optimizer_type:
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


@job(name="optimize_prompts", max_workers=_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS)
def optimize_prompts_job(
    run_id: str,
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
    It resumes an existing MLflow run (created by the handler) and calls
    mlflow.genai.optimize_prompts() which reuses the active run.

    Note: This job only supports single-prompt optimization. The predict_fn
    is automatically built using the prompt's model_config (provider/model_name)
    via litellm, making the optimization self-contained without requiring users
    to serialize their own predict function.

    Args:
        run_id: The MLflow run ID to resume (created by the handler upfront).
        experiment_id: The experiment ID to track the optimization in.
        prompt_uri: The URI of the prompt to optimize.
        dataset_id: The ID of the EvaluationDataset containing training data.
        optimizer_type: The optimizer type string (e.g., "gepa", "metaprompt").
        optimizer_config_json: JSON string of optimizer-specific configuration.
        scorers: List of scorer names. Can be built-in scorer class names
            (e.g., "Correctness", "Safety") or registered scorer names from
            the experiment's scorer registry.

    Returns:
        Dict containing optimization results:
        - run_id: The MLflow run ID
        - source_prompt_uri: URI of the source prompt
        - optimized_prompt_uri: URI of the optimized prompt
        - optimizer_name: Name of the optimizer used
        - initial_eval_score: Initial evaluation score (if available)
        - final_eval_score: Final evaluation score (if available)
    """
    # Record telemetry event for job execution
    _record_event(
        OptimizePromptsJobEvent,
        {
            "optimizer_type": optimizer_type,
            "scorer_count": len(scorers),
        },
    )

    set_experiment(experiment_id=experiment_id)

    dataset = get_dataset(dataset_id=dataset_id)
    # Convert to DataFrame since optimize_prompts expects a len()-able object
    train_data = dataset.to_df()
    _logger.info(f"Loaded dataset {dataset_id} with {len(train_data)} samples")

    predict_fn = _build_predict_fn(prompt_uri)
    _logger.info(f"Built predict_fn from prompt {prompt_uri}")

    optimizer = _create_optimizer(optimizer_type, optimizer_config_json)
    _logger.info(f"Created optimizer {optimizer_type}")

    loaded_scorers = _load_scorers(scorers, experiment_id)
    _logger.info(f"Loaded {len(loaded_scorers)} scorers: {scorers}")

    source_prompt = load_prompt(prompt_uri)
    _logger.info(f"Loaded source prompt: {prompt_uri}")

    # Resume the given run ID. Params have already been logged by the handler
    with start_run(run_id=run_id):
        # Link source prompt to run for lineage
        client = MlflowClient()
        client.link_prompt_version_to_run(run_id=run_id, prompt=source_prompt)
        _logger.info(f"Linked source prompt {prompt_uri} to run {run_id}")

        # Run optimization - reuses this active run
        result = optimize_prompts(
            predict_fn=predict_fn,
            train_data=train_data,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=loaded_scorers,
            enable_tracking=True,
        )

    return {
        "run_id": run_id,
        "source_prompt_uri": prompt_uri,
        "optimized_prompt_uri": result.optimized_prompts[0].uri
        if result.optimized_prompts
        else None,
        "optimizer_name": result.optimizer_name,
        "initial_eval_score": result.initial_eval_score,
        "final_eval_score": result.final_eval_score,
        "dataset_id": dataset_id,
        "scorers": scorers,
    }
