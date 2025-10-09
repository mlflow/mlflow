import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

import mlflow
from mlflow.entities.model_registry import PromptVersion
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.utils import (
    _convert_eval_set_to_df,
)
from mlflow.genai.optimize.adapters import BasePromptAdapter, get_default_adapter
from mlflow.genai.optimize.types import EvaluationResultRecord, LLMParams, PromptAdaptationResult
from mlflow.genai.prompts import load_prompt, register_prompt
from mlflow.genai.utils.trace_utils import convert_predict_fn
from mlflow.telemetry.events import PromptAdaptationEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils import gorilla
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils.safety import _wrap_patch

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes


@experimental(version="3.5.0")
@record_usage_event(PromptAdaptationEvent)
def adapt_prompts(
    predict_fn: Callable[..., Any],
    train_data: "EvaluationDatasetTypes",
    target_prompt_uris: list[str],
    optimizer_lm_params: LLMParams,
    optimizer: BasePromptAdapter | None = None,
) -> PromptAdaptationResult:
    """
    This API optimizes prompts used in the passed in function to produce similar
    outputs as the outputs in the dataset. This API can be used to maintain the
    outputs of `predict_fn` when the language model used in the function is changed.

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
        optimizer_lm_params: model parameters used in the optimization algorithm.
            The model name can be specified in either format:
            - `<provider>:/<model>` (e.g., "openai:/gpt-4o")
            - `<provider>/<model>` (e.g., "openai/gpt-4o")
        optimizer: an optional prompt optimizer object that optimizes a set of prompts based
            on the evaluation dataset and passed in function.
            If this argument is none, the default optimizer is used.

    Returns:
        A list of optimized prompt versions.

    Examples:

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize import LLMParams

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

            result = mlflow.genai.adapt_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                target_prompt_uris=[prompt.uri],
                optimizer_lm_params=LLMParams(model_name="openai:/gpt-4o"),
            )

            print(result.optimized_prompts[0].template)
    """
    if optimizer is None:
        optimizer = get_default_adapter()

    # TODO: Add dataset validation
    converted_train_data = _convert_eval_set_to_df(train_data).to_dict("records")
    predict_fn = convert_predict_fn(
        predict_fn=predict_fn, sample_input=converted_train_data[0]["inputs"]
    )

    eval_fn = _build_eval_fn(predict_fn)

    target_prompts = [load_prompt(prompt_uri) for prompt_uri in target_prompt_uris]
    target_prompts_dict = {prompt.name: prompt.template for prompt in target_prompts}

    optimizer_output = optimizer.optimize(
        eval_fn, converted_train_data, target_prompts_dict, optimizer_lm_params
    )

    return PromptAdaptationResult(
        optimized_prompts=[
            register_prompt(name=prompt_name, template=prompt)
            for prompt_name, prompt in optimizer_output.optimized_prompts.items()
        ],
        optimizer_name=optimizer.__class__.__name__,
    )


def _build_eval_fn(
    predict_fn: Callable[..., Any],
) -> Callable[[dict[str, str], list[dict[str, Any]]], list[EvaluationResultRecord]]:
    """
    Build an evaluation function that uses the candidate prompts to evaluate the predict_fn.
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
            outputs = record["outputs"]
            eval_request_id = str(uuid.uuid4())
            # set prediction context to retrieve the trace by the request id,
            # and set is_evaluate to True to disable async trace logging
            with set_prediction_context(Context(request_id=eval_request_id, is_evaluate=True)):
                try:
                    program_outputs = predict_fn(inputs)
                except Exception as e:
                    program_outputs = f"Failed to invoke the predict_fn with {inputs}: {e}"

            trace = mlflow.get_trace(eval_request_id, silent=True)
            # TODO: Consider more robust scoring mechanism
            score = 1 if program_outputs == outputs else 0
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
                thread_name_prefix="MlflowGenAIEvalHarness",
            ) as executor:
                futures = [executor.submit(_run_single, record) for record in dataset]
                return [future.result() for future in futures]
        finally:
            gorilla.revert(patch)

    return eval_fn
