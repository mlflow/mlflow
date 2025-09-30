from typing import Callable, Any, TYPE_CHECKING
import uuid

import mlflow
from mlflow.entities.model_registry import PromptVersion
from mlflow.genai.optimize.adapters import BasePromptAdapter, get_default_adapter
from mlflow.genai.optimize.types import EvaluationResultRecord, LLMParams
from mlflow.genai.prompts import load_prompt, register_prompt
from mlflow.genai.evaluation.utils import (
    _convert_eval_set_to_df,
)
from mlflow.genai.utils.trace_utils import convert_predict_fn
from mlflow.pyfunc import Context, set_prediction_context
from mlflow.utils import gorilla
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils.safety import _wrap_patch

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes

@experimental(version="3.5.0")
def adapt_prompts(
  predict_fn: Callable[..., Any],
  train_data: "EvaluationDatasetTypes",
  target_prompt_uris: list[str],
  optimizer_lm_params: LLMParams,
  oprimizer: BasePromptAdapter | None = None,
) -> list[PromptVersion]:
    """
    This API optimizes prompts used in the passed in function to produce similar outputs as the outputs in the dataset.
    This API can be used to maintain the outputs of `predict_fn` when the language model used in the function is changed.

    Args:
        predict_fn: a target function to be optimized. The callable should receive inputs as keyword arguments and return the response. 
            The function should use MLflow prompt registry and call `PromptVersion.format` during execution in order for this API to optimize the prompt.
        train_data: an evaluation dataset used for the optimization. It should include the inputs and outputs fields with dict values.
            The data must be one of the following formats:

            * An EvaluationDataset entity
            * Pandas DataFrame
            * Spark DataFrame
            * List of dictionaries

            The dataset must include the following columns:

            - inputs: A column containing single inputs in dict format.
              Each input should contain keys matching the variables in the prompt template.
            - outputs: A column containing an output for each input that the predict_fn should produce.
        target_prompt_uris: a list of prompt uris to be optimized. The prompt templates should be used by the predict_fn.
        optimizer_lm_params: model parameters used in the optimization algorithm.
            The model name can be specified in either format:
            - `<provider>:/<model>` (e.g., "openai:/gpt-4o")
            - `<provider>/<model>` (e.g., "openai/gpt-4o")
        oprimizer: an optional prompt optimizer object that optimizes a set of prompts based on the evaluation dataset and passed in function. 
            If this argument is none, the default optimizer is used.

    Returns:
        A list of optimized prompt versions.
    """
    if oprimizer is None:
        oprimizer = get_default_adapter()
    
    # TODO: Add dataset validation
    converted_train_data = _convert_eval_set_to_df(train_data).to_dict("records")
    predict_fn = convert_predict_fn(predict_fn=predict_fn, sample_input=converted_train_data[0]["inputs"])

    eval_fn = _build_eval_fn(predict_fn)
    
    target_prompts = [load_prompt(prompt_uri) for prompt_uri in target_prompt_uris]
    target_prompts_dict = {prompt.name: prompt.template for prompt in target_prompts}

    prompts = oprimizer.optimize(eval_fn, converted_train_data, target_prompts_dict, optimizer_lm_params)

    return [register_prompt(name=prompt_name, template=prompt) for prompt_name, prompt in prompts.items()]

def _build_eval_fn(predict_fn: Callable[..., Any]) -> Callable[[dict[str, str], list[dict[str, Any]]], list[EvaluationResultRecord]]:
    """
    Build an evaluation function that uses the candidate prompts to evaluate the predict_fn.
    """
    def eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]) -> list[EvaluationResultRecord]:
        original_format_fn = PromptVersion.format
        def _prompt_format_patch(self, **kwargs: Any) -> str:
            """
            Patch the format method of PromptVersion to return the candidate prompt if it is in the candidate_prompts dict.
            """
            template_name = self.name
            if template_name in candidate_prompts:
                return candidate_prompts[template_name]
            return original_format_fn(self, **kwargs)
        results = []
        patch = _wrap_patch(PromptVersion, "format", _prompt_format_patch)
        try:
            for record in dataset:
                inputs = record["inputs"]
                outputs = record["outputs"]
                eval_request_id = str(uuid.uuid4())
                # set prediction context to retrieve the trace by the request id, and set is_evaluate to True to disable async trace logging
                with set_prediction_context(Context(request_id=eval_request_id, is_evaluate=True)):
                    try:
                        target_outputs = predict_fn(inputs)
                    except Exception as e:
                        print(f"Failed to invoke the predict_fn with {inputs}: {e}")
                        target_outputs = (
                            f"Failed to invoke the predict_fn with {inputs}: {e}"
                        )

                trace = mlflow.get_trace(eval_request_id, silent=True)
                # TODO: Consider more robust scoring mechanism
                score = 1 if target_outputs == outputs else 0
                results.append(EvaluationResultRecord(inputs=inputs, outputs=target_outputs, score=score, trace=trace))
        finally:
            gorilla.revert(patch)
        return results
    
    return eval_fn