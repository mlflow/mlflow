import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

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
from mlflow.protos.prompt_optimization_pb2 import (
    OPTIMIZER_TYPE_GEPA,
    OPTIMIZER_TYPE_METAPROMPT,
    OPTIMIZER_TYPE_UNSPECIFIED,
)
from mlflow.protos.prompt_optimization_pb2 import OptimizerType as ProtoOptimizerType
from mlflow.server.jobs import job
from mlflow.telemetry.events import OptimizePromptsJobEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import set_experiment, start_run

_logger = logging.getLogger(__name__)

_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS = 2


class OptimizerType(str, Enum):
    """Supported prompt optimizer types."""

    GEPA = "gepa"
    METAPROMPT = "metaprompt"

    @classmethod
    def from_proto(cls, proto_value: int) -> "OptimizerType":
        """
        Convert a proto OptimizerType enum value to the Python OptimizerType enum.

        Args:
            proto_value: The integer value from the proto OptimizerType enum.

        Returns:
            The corresponding OptimizerType enum member.

        Raises:
            MlflowException: If the proto value is unspecified or unsupported.
        """
        if proto_value == OPTIMIZER_TYPE_UNSPECIFIED:
            supported_types = [
                name for name in ProtoOptimizerType.keys() if name != "OPTIMIZER_TYPE_UNSPECIFIED"
            ]
            raise MlflowException.invalid_parameter_value(
                f"optimizer_type is required. Supported types: {supported_types}"
            )
        elif proto_value == OPTIMIZER_TYPE_GEPA:
            return cls.GEPA
        elif proto_value == OPTIMIZER_TYPE_METAPROMPT:
            return cls.METAPROMPT
        else:
            supported_types = [
                name for name in ProtoOptimizerType.keys() if name != "OPTIMIZER_TYPE_UNSPECIFIED"
            ]
            raise MlflowException.invalid_parameter_value(
                f"Unsupported optimizer_type value: {proto_value}. "
                f"Supported types: {supported_types}"
            )

    def to_proto(self) -> int:
        """
        Convert the Python OptimizerType enum to a proto OptimizerType enum value.

        Returns:
            The corresponding proto OptimizerType integer value.
        """
        if self == OptimizerType.GEPA:
            return OPTIMIZER_TYPE_GEPA
        elif self == OptimizerType.METAPROMPT:
            return OPTIMIZER_TYPE_METAPROMPT
        return OPTIMIZER_TYPE_UNSPECIFIED


@dataclass
class PromptOptimizationJobResult:
    run_id: str
    source_prompt_uri: str
    optimized_prompt_uri: str | None
    optimizer_name: str
    initial_eval_score: float | None
    final_eval_score: float | None
    dataset_id: str
    scorer_names: list[str]


def _create_optimizer(
    optimizer_type: str,
    optimizer_config: dict[str, Any] | None,
) -> BasePromptOptimizer:
    """
    Create an optimizer instance from type string and configuration dict.

    Args:
        optimizer_type: The optimizer type string (e.g., "gepa", "metaprompt").
        optimizer_config: Optimizer-specific configuration dictionary.

    Returns:
        An instantiated optimizer.

    Raises:
        MlflowException: If optimizer type is not supported.
    """
    config = optimizer_config or {}
    optimizer_type_lower = optimizer_type.lower() if optimizer_type else ""

    if optimizer_type_lower == OptimizerType.GEPA:
        reflection_model = config.get("reflection_model")
        if not reflection_model:
            raise MlflowException.invalid_parameter_value(
                "Missing required optimizer configuration: 'reflection_model' must be specified "
                "in optimizer_config for the GEPA optimizer (e.g., 'openai:/gpt-4o')."
            )
        return GepaPromptOptimizer(
            reflection_model=reflection_model,
            max_metric_calls=config.get("max_metric_calls", 100),
            display_progress_bar=config.get("display_progress_bar", False),
            gepa_kwargs=config.get("gepa_kwargs"),
        )
    elif optimizer_type_lower == OptimizerType.METAPROMPT:
        reflection_model = config.get("reflection_model")
        if not reflection_model:
            raise MlflowException.invalid_parameter_value(
                "Missing required optimizer configuration: 'reflection_model' must be specified "
                "in optimizer_config for the MetaPrompt optimizer (e.g., 'openai:/gpt-4o')."
            )
        return MetaPromptOptimizer(
            reflection_model=reflection_model,
            lm_kwargs=config.get("lm_kwargs"),
            guidelines=config.get("guidelines"),
        )
    elif not optimizer_type:
        supported_types = [t.value for t in OptimizerType]
        raise MlflowException.invalid_parameter_value(
            f"Optimizer type must be specified. Supported types: {supported_types}"
        )
    else:
        supported_types = [t.value for t in OptimizerType]
        raise MlflowException.invalid_parameter_value(
            f"Unsupported optimizer type: '{optimizer_type}'. Supported types: {supported_types}"
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
        scorer_class = getattr(builtin_scorers, name, None)
        if scorer_class is not None:
            try:
                scorer = scorer_class()
                scorers.append(scorer)
                continue
            except Exception as e:
                _logger.debug(f"Failed to instantiate built-in scorer {name}: {e}")

        # Load from the registered scorer store if not a built-in scorer
        try:
            scorer = get_scorer(name=name, experiment_id=experiment_id)
            scorers.append(scorer)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Scorer '{name}' not found. It is neither a built-in scorer "
                f"(e.g., 'Correctness', 'Safety') nor a registered scorer in "
                f"experiment '{experiment_id}'. Error: {e}"
            )

    return scorers


def _build_predict_fn(prompt_uri: str) -> Callable[..., Any]:
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
    try:
        import litellm
    except ImportError as e:
        raise MlflowException(
            "The 'litellm' package is required for prompt optimization but is not installed. "
            "Please install it using: pip install litellm"
        ) from e

    prompt = load_prompt(prompt_uri)
    try:
        model_config = prompt.model_config
        provider = model_config["provider"]
        model_name = model_config["model_name"]
    except (KeyError, TypeError, AttributeError) as e:
        raise MlflowException(
            f"Prompt {prompt_uri} doesn't have a model configuration that sets provider and "
            "model_name, which are required for optimization."
        ) from e

    litellm_model = f"{provider}/{model_name}"

    def predict_fn(**kwargs: Any) -> Any:
        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt.format(**kwargs)}],
        )
        return response.choices[0].message.content

    return predict_fn


@record_usage_event(OptimizePromptsJobEvent)
@job(name="optimize_prompts", max_workers=_DEFAULT_OPTIMIZATION_JOB_MAX_WORKERS)
def optimize_prompts_job(
    run_id: str,
    experiment_id: str,
    prompt_uri: str,
    dataset_id: str,
    optimizer_type: str,
    optimizer_config: dict[str, Any] | None,
    scorer_names: list[str],
) -> dict[str, Any]:
    """
    Job function for async single-prompt optimization.

    This function is executed as a background job by the MLflow server.
    It resumes an existing MLflow run (created by the handler) and calls
    `mlflow.genai.optimize_prompts()` which reuses the active run.

    Note: This job only supports single-prompt optimization. The predict_fn
    is automatically built using the prompt's model_config (provider/model_name)
    via litellm, making the optimization self-contained without requiring users
    to serialize their own predict function.

    Args:
        run_id: The MLflow run ID to track the optimization configs and metrics.
        experiment_id: The experiment ID to track the optimization in.
        prompt_uri: The URI of the prompt to optimize.
        dataset_id: The ID of the EvaluationDataset containing training data.
        optimizer_type: The optimizer type string (e.g., "gepa", "metaprompt").
        optimizer_config: Optimizer-specific configuration dictionary.
        scorer_names: List of scorer names. Can be built-in scorer class names
            (e.g., "Correctness", "Safety") or registered scorer names.
            For custom scorers, use mlflow.genai.make_judge() to create a judge,
            then register it using scorer.register(experiment_id=experiment_id),
            and pass the registered scorer name here.

    Returns:
        Dict containing optimization results and metadata (JSON-serializable).
    """
    set_experiment(experiment_id=experiment_id)

    dataset = get_dataset(dataset_id=dataset_id) if dataset_id else None
    predict_fn = _build_predict_fn(prompt_uri)
    optimizer = _create_optimizer(optimizer_type, optimizer_config)
    loaded_scorers = _load_scorers(scorer_names, experiment_id)
    source_prompt = load_prompt(prompt_uri)

    # Resume the given run ID. Params have already been logged by the handler
    with start_run(run_id=run_id):
        # Link source prompt to run for lineage
        client = MlflowClient()
        client.link_prompt_version_to_run(run_id=run_id, prompt=source_prompt)
        result = optimize_prompts(
            predict_fn=predict_fn,
            train_data=dataset,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=loaded_scorers,
            enable_tracking=True,
        )

    job_result = PromptOptimizationJobResult(
        run_id=run_id,
        source_prompt_uri=prompt_uri,
        optimized_prompt_uri=result.optimized_prompts[0].uri if result.optimized_prompts else None,
        optimizer_name=result.optimizer_name,
        initial_eval_score=result.initial_eval_score,
        final_eval_score=result.final_eval_score,
        dataset_id=dataset_id,
        scorer_names=scorer_names,
    )
    return asdict(job_result)
