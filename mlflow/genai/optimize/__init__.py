from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimize import optimize_prompts
from mlflow.genai.optimize.optimizers import BasePromptOptimizer, GepaPromptOptimizer
from mlflow.genai.optimize.types import (
    LLMParams,
    OptimizerConfig,
    PromptOptimizationResult,
    PromptOptimizerOutput,
)

_MIGRATION_GUIDE = """
    Migration guide:
        The ``optimize_prompt()`` API has been replaced by
        :py:func:`mlflow.genai.optimize_prompts()`, which provides more flexible
        optimization capabilities with a joint optimization of prompts in an arbitrary function.

        **Old API (removed):**

        .. code-block:: python

            from mlflow.genai import optimize_prompt
            from mlflow.genai.optimize.types import OptimizerConfig, LLMParams

            result = optimize_prompt(
                target_llm=LLMParams(model_name="openai:/gpt-4o"),
                prompt="prompts:/my-prompt/1",
                train_data=dataset,
                optimizer_config=OptimizerConfig(num_instruction_candidates=10),
            )

        **New API:**

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai import optimize_prompts
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
            from mlflow.genai.scorers import Correctness


            # Define a predict function that uses the prompt and LLM
            def predict_fn(inputs: dict[str, Any]) -> str:
                prompt = mlflow.genai.load_prompt("prompts:/my-prompt/1")
                formatted_prompt = prompt.format(**inputs)
                completion = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": formatted_prompt}],
                )
                return completion.choices[0].message.content


            result = optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                prompt_uris=["prompts:/my-prompt/1"],
                optimizer=GepaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    max_metric_calls=100,
                ),
                scorers=[Correctness(model="openai:/gpt-4o")],
            )

        Key differences:
        - Use ``optimize_prompts()`` (plural) instead of ``optimize_prompt()``
        - Provide a predict function ``predict_fn`` instead of a prompt uri ``prompt``
        - Use ``prompt_uris`` to reference registered prompts
        - Specify an ``optimizer`` instance (e.g., ``GepaPromptOptimizer``)

        For more details, see the documentation:
        https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts.html
        """


def optimize_prompt(*args, **kwargs):
    f"""
    Optimize a LLM prompt using the given dataset and evaluation metrics.
    This function has been removed. Use mlflow.genai.optimize_prompts() instead.

    {_MIGRATION_GUIDE}
    """

    raise MlflowException(
        f"""
        The optimize_prompt() function has been removed in MLflow 3.5.0.
        Please use mlflow.genai.optimize_prompts() instead.
        {_MIGRATION_GUIDE}"""
    )


__all__ = [
    "optimize_prompts",
    "optimize_prompt",
    "LLMParams",
    "OptimizerConfig",
    "BasePromptOptimizer",
    "GepaPromptOptimizer",
    "PromptOptimizerOutput",
    "PromptOptimizationResult",
]
