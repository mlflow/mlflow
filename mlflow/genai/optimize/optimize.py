import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

import mlflow
from mlflow.entities.model_registry import PromptVersion
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.utils import (
    _convert_eval_set_to_df,
)
from mlflow.genai.optimize.optimizers import BasePromptOptimizer
from mlflow.genai.optimize.types import (
    EvaluationResultRecord,
    ObjectiveFn,
    PromptOptimizationResult,
)
from mlflow.genai.optimize.util import create_metric_from_scorers
from mlflow.genai.prompts import load_prompt, register_prompt
from mlflow.genai.scorers import Scorer, scorer
from mlflow.genai.utils.trace_utils import convert_predict_fn
from mlflow.telemetry.events import PromptOptimizationEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils import gorilla
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils.safety import _wrap_patch

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
@record_usage_event(PromptOptimizationEvent)
def optimize_prompts(
    *,
    predict_fn: Callable[..., Any],
    train_data: "EvaluationDatasetTypes",
    target_prompt_uris: list[str],
    optimizer: BasePromptOptimizer,
    scorers: list[Scorer] | None = None,
    objective: ObjectiveFn | None = None,
) -> PromptOptimizationResult:
    """
    Automatically optimize prompts using evaluation metrics and training data.

    This function uses optimization algorithms (such as GEPA) to improve prompt
    quality based on your evaluation criteria. It's ideal for:

    - Improving prompt performance on specific tasks
    - Adapting prompts when switching between language models
    - Systematically enhancing prompt quality with data-driven techniques

    Args:
        predict_fn: a target function to be optimized. The callable should receive inputs
            as keyword arguments and return the response. The function should use
            MLflow prompt registry and call `PromptVersion.format` during execution
            in order for this API to optimize the prompt. This function should return the
            same type as the outputs in the dataset.
        train_data: an evaluation dataset used for the optimization.
            It should include the inputs and outputs fields with dict values.
            The data must be one of the following formats:

            * An EvaluationDataset entity
            * Pandas DataFrame
            * Spark DataFrame
            * List of dictionaries

            The dataset must include the following columns:

            - inputs: A column containing single inputs in dict format.
              Each input should contain keys matching the variables in the prompt template.
            - outputs: A column containing an output for each input
              that the predict_fn should produce.
        target_prompt_uris: a list of prompt uris to be optimized.
            The prompt templates should be used by the predict_fn.
        optimizer: a prompt optimizer object that optimizes a set of prompts based
            on the evaluation dataset and passed in function. For example,
            GepaPromptOptimizer(reflection_model="openai:/gpt-4o").
        scorers: List of scorers that evaluate the inputs, outputs and expectations.
            If None, uses a default scorer that applies exact match for numeric types
            and LLM judge for text.
        objective: A callable that computes the overall performance metric from individual
            scorer outputs. Takes a dict mapping scorer names to scores and returns a float
            value (greater is better). If None and all scorers return numerical values,
            uses sum of scores by default.

    Returns:
        The optimization result object that includes the optimized prompts
        as a list of prompt versions and the optimizer name.

    Examples:

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

            prompt = mlflow.genai.register_prompt(
                name="qa",
                template="Answer the following question: {{question}}",
            )


            def predict_fn(question: str) -> str:
                completion = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt.format(question=question)}],
                )
                return completion.choices[0].message.content


            dataset = [
                {"inputs": {"question": "What is the capital of France?"}, "outputs": "Paris"},
                {"inputs": {"question": "What is the capital of Germany?"}, "outputs": "Berlin"},
            ]

            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                target_prompt_uris=[prompt.uri],
                optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o"),
            )

            print(result.optimized_prompts[0].template)

        **Example: Using custom scorers with an objective function**

        .. code-block:: python

            import mlflow
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
            from mlflow.genai.scorers import scorer


            # Define custom scorers
            @scorer(name="accuracy")
            def accuracy_scorer(outputs, expectations):
                return 1.0 if outputs.lower() == expectations.lower() else 0.0


            @scorer(name="brevity")
            def brevity_scorer(outputs):
                # Prefer shorter outputs (max 50 chars gets score of 1.0)
                return min(1.0, 50 / max(len(outputs), 1))


            # Define objective to combine scores
            def weighted_objective(scores):
                return 0.7 * scores["accuracy"] + 0.3 * scores["brevity"]


            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                target_prompt_uris=[prompt.uri],
                optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o"),
                scorers=[accuracy_scorer, brevity_scorer],
                objective=weighted_objective,
            )
    """
    # Use default scorer if none provided
    if not scorers:
        scorers = [_make_output_equivalence_scorer(optimizer.model_name)]

    # TODO: Add dataset validation
    converted_train_data = _convert_eval_set_to_df(train_data).to_dict("records")
    predict_fn = convert_predict_fn(
        predict_fn=predict_fn, sample_input=converted_train_data[0]["inputs"]
    )

    metric_fn = create_metric_from_scorers(scorers, objective)
    eval_fn = _build_eval_fn(predict_fn, metric_fn)

    target_prompts = [load_prompt(prompt_uri) for prompt_uri in target_prompt_uris]
    target_prompts_dict = {prompt.name: prompt.template for prompt in target_prompts}

    optimizer_output = optimizer.optimize(eval_fn, converted_train_data, target_prompts_dict)

    return PromptOptimizationResult(
        optimized_prompts=[
            register_prompt(name=prompt_name, template=prompt)
            for prompt_name, prompt in optimizer_output.optimized_prompts.items()
        ],
        optimizer_name=optimizer.__class__.__name__,
        initial_eval_score=optimizer_output.initial_eval_score,
        final_eval_score=optimizer_output.final_eval_score,
    )


def _build_eval_fn(
    predict_fn: Callable[..., Any],
    metric_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float],
) -> Callable[[dict[str, str], list[dict[str, Any]]], list[EvaluationResultRecord]]:
    """
    Build an evaluation function that uses the candidate prompts to evaluate the predict_fn.

    Args:
        predict_fn: The function to evaluate
        metric_fn: Metric function created from scorers that takes (inputs, outputs, expectations)

    Returns:
        An evaluation function
    """
    from mlflow.pyfunc import Context, set_prediction_context

    def eval_fn(
        candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]
    ) -> list[EvaluationResultRecord]:
        @property
        def _template_patch(self) -> str:
            template_name = self.name
            if template_name in candidate_prompts:
                return candidate_prompts[template_name]
            return self.template

        patch = _wrap_patch(PromptVersion, "template", _template_patch)

        def _run_single(record: dict[str, Any]):
            inputs = record["inputs"]
            # use expectations if provided, otherwise use outputs
            outputs = record.get("expectations") or record.get("outputs")
            eval_request_id = str(uuid.uuid4())
            # set prediction context to retrieve the trace by the request id,
            # and set is_evaluate to True to disable async trace logging
            with set_prediction_context(Context(request_id=eval_request_id, is_evaluate=True)):
                try:
                    program_outputs = predict_fn(inputs)
                except Exception as e:
                    program_outputs = f"Failed to invoke the predict_fn with {inputs}: {e}"

            trace = mlflow.get_trace(eval_request_id, silent=True)
            # Use metric function created from scorers
            score = metric_fn(inputs=inputs, outputs=program_outputs, expectations=outputs)
            return EvaluationResultRecord(
                inputs=inputs,
                outputs=program_outputs,
                expectations=outputs,
                score=score,
                trace=trace,
            )

        try:
            with ThreadPoolExecutor(
                max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
                thread_name_prefix="MLflowPromptAdaptation",
            ) as executor:
                futures = [executor.submit(_run_single, record) for record in dataset]
                return [future.result() for future in futures]
        finally:
            gorilla.revert(patch)

    return eval_fn


def _make_output_equivalence_scorer(judge_model: str) -> Scorer:
    """
    Create an output equivalence scorer with a specific judge model.

    Args:
        judge_model: The model to use for LLM judge evaluation

    Returns:
        A Scorer that compares outputs against expected outputs
    """

    @scorer(name="output_equivalence")
    def output_equivalence(outputs: Any, expectations: Any) -> float:
        """
        Compare outputs against expected outputs.

        Uses exact match for numerical types and LLM judge for text types.

        Args:
            outputs: The actual output from the program
            expectations: The expected output to match

        Returns:
            A score between 0 and 1
        """
        from mlflow.genai.judges import make_judge

        # Handle exact match for numerical types
        if isinstance(outputs, (int, float, bool)) and isinstance(expectations, (int, float, bool)):
            return 1.0 if outputs == expectations else 0.0

        # Convert to strings for comparison
        outputs_str = str(outputs)
        expectations_str = str(expectations)

        # Use exact match first
        if outputs_str == expectations_str:
            return 1.0

        # Use LLM judge for text outputs
        judge = make_judge(
            name="equivalence_judge",
            instructions=(
                "Compare {{outputs}} against {{expectations}}. "
                "Evaluate if they are both semantically equivalent or convey the same meaning, "
                "and if the output format matches the expected format "
                "(e.g., JSON structure, list format, sentence structure). "
                "Return 'pass' if they match in both content and format, 'fail' if they don't."
            ),
            model=judge_model,
        )
        try:
            result = judge(
                outputs={"outputs": outputs_str}, expectations={"outputs": expectations_str}
            )
            result_value = str(result.value).lower().strip()
            return 1.0 if result_value == "pass" else 0.0
        except Exception as e:
            _logger.warning("Failed to compute score with LLM judge: %s", e)
            return 0.0

    return output_equivalence
