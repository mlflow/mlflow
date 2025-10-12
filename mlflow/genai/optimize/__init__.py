from mlflow.genai.optimize.adapt import adapt_prompts
from mlflow.genai.optimize.adapters import BasePromptAdapter
from mlflow.genai.optimize.types import (
    EvaluationResultRecord,
    LLMParams,
    OptimizerConfig,
    PromptAdaptationResult,
    PromptAdapterOutput,
)
from mlflow.utils.annotations import deprecated


@deprecated(
    since="3.5.0",
    impact="This function has been removed. Use mlflow.genai.optimize_prompts() instead.",
)
def optimize_prompt(*args, **kwargs):
    """
    Optimize prompts for a given task.

    .. deprecated:: 3.5.0
        ``optimize_prompt()`` has been removed. Use :py:func:`mlflow.genai.optimize_prompts()` instead.

    Migration guide:
        The ``optimize_prompt()`` API has been replaced by :py:func:`mlflow.genai.optimize_prompts()`,
        which provides a more flexible and intuitive interface.

        **Old API (removed):**

        .. code-block:: python

            from mlflow.genai import optimize_prompt
            from mlflow.genai.optimize.types import OptimizerConfig, LLMParams

            result = optimize_prompt(
                target_llm=LLMParams(model_name="openai:/gpt-4o"),
                initial_prompt="Answer the question: {{question}}",
                train_data=dataset,
                config=OptimizerConfig(num_instruction_candidates=10),
            )

        **New API:**

        .. code-block:: python

            from mlflow.genai import optimize_prompts
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
            from mlflow.genai.scorers.builtin_scorers import OutputEquivalence

            result = optimize_prompts(
                predict_fn=lambda inputs: model.predict(inputs),
                train_data=dataset,
                prompt_uris=["prompts:/my-prompt@candidate"],
                optimizer=GepaPromptOptimizer(
                    model="openai:/gpt-4o",
                    reflection_model="openai:/gpt-4o",
                    num_iterations=10,
                ),
                scorers=[OutputEquivalence()],
            )

        Key differences:
        - Use ``optimize_prompts()`` (plural) instead of ``optimize_prompt()``
        - Provide a ``predict_fn`` instead of ``target_llm``
        - Use ``prompt_uris`` to reference registered prompts
        - Specify an ``optimizer`` instance (e.g., ``GepaPromptOptimizer``)
        - Pass ``scorers`` to define evaluation metrics

        For more details, see the documentation:
        https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts.html
    """
    from mlflow.exceptions import MlflowException

    raise MlflowException(
        "The optimize_prompt() function has been removed in MLflow 3.5.0. "
        "Please use mlflow.genai.optimize_prompts() instead. "
        "See the function docstring for migration instructions."
    )


__all__ = [
    "adapt_prompts",
    "optimize_prompt",
    "EvaluationResultRecord",
    "LLMParams",
    "OptimizerConfig",
    "BasePromptAdapter",
    "PromptAdapterOutput",
    "PromptAdaptationResult",
]
