from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal, Optional, Union

import pandas as pd
import pydantic

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer, scorer
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.prompt_template import PromptTemplate
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    UNAUTHENTICATED,
    ErrorCode,
)


def _cast_to_result_type(raw_result: str, result_type: Union[type, pydantic.BaseModel]):
    """
    Cast the raw string output from an LLM to the specified result type.

    Args:
        raw_result: String output from the LLM
        result_type: Target type to convert to (primitive type or Pydantic model)

    Returns:
        Converted value matching the specified result_type
    """

    # Handle primitive types
    if result_type == bool:
        lower_result = raw_result.lower().strip()
        if lower_result in ("true", "yes", "1", "t", "y"):
            return True
        elif lower_result in ("false", "no", "0", "f", "n"):
            return False
        else:
            return None
    elif result_type == int:
        try:
            return int(raw_result.strip())
        except (ValueError, TypeError):
            return None
    elif result_type == float:
        try:
            return float(raw_result.strip())
        except (ValueError, TypeError):
            return None
    elif result_type == str:
        return raw_result

    # TODO: add support for pydantic models

    return None


def _score_model_on_one_payload(
    payload: str,
    eval_model: str,
    result_type: Union[type, pydantic.BaseModel],
    parameters: Optional[dict[str, Any]],
):
    try:
        # If the endpoint does not specify type, default to chat format
        endpoint_type = model_utils.get_endpoint_type(eval_model) or "llm/v1/chat"
        raw_result = model_utils.score_model_on_payload(
            eval_model, payload, parameters, endpoint_type
        )
        return _cast_to_result_type(raw_result, result_type)
    except ImportError:
        raise
    except MlflowException as e:
        if e.error_code in [
            ErrorCode.Name(BAD_REQUEST),
            ErrorCode.Name(UNAUTHENTICATED),
            ErrorCode.Name(INVALID_PARAMETER_VALUE),
        ]:
            raise
        else:
            return None, f"Failed to score model on payload. Error: {e!s}"
    except Exception as e:
        return None, f"Failed to score model on payload. Error: {e!s}"


def _score_model_on_payloads(
    grading_payloads, model, parameters, result_type, max_workers
) -> list[Union[int, float, bool, str]]:
    scores = [None] * len(grading_payloads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_model_on_one_payload,
                payload,
                model,
                result_type,
                parameters,
            ): indx
            for indx, payload in enumerate(grading_payloads)
        }

        as_comp = as_completed(futures)
        try:
            from tqdm.auto import tqdm

            as_comp = tqdm(as_comp, total=len(futures))
        except ImportError:
            pass

        for future in as_comp:
            indx = futures[future]
            score = future.result()
            scores[indx] = score

    return scores


def llm_judge_scorer(
    name: str,
    prompt_template: str,
    judge: str = "openai:/gpt-4o",
    result_type: type = bool,
    max_workers: int = 10,
    aggregations: Optional[
        list[Union[Literal["max", "min", "mean", "p90", "p99"], Callable]]
    ] = None,
) -> Scorer:
    """
    Define a scorer that produces an assessment via LLM-as-a-Judge.

    Args:
        name: The name of the assessment.
        prompt_template: The prompt template passed to the judge. You can use variables
            wrapped with double-curly braces e.g. {{ outputs }} to fill-in any fields from
            the input dataset into prompts. There are a few reserved variable names:

                - inputs: The 'inputs' column in the dataset.
                - outputs: The 'outputs' column in the dataset.
        judge: The judge model to score the assessment. Specify model provider and a model
            name in '<provider>:/<model>' format. Default is OpenAI GPT4o (tentative)
        result_type: A type of judge outputs.
        max_workers: Number of max parallelization to make LLM requests.
        aggregations: The list of options to aggregate the scores. Supported
            options are: min, max, mean, median, variance, p90.
            To use a custom aggregation, specify a function that takes a list of
            Assessment and outputs a Metric.
    """
    # TODO: aggregations require updating databricks-agent eval
    # TODO: switch to structured generation API instead of optimistically casting to result_type

    prompt_template = PromptTemplate(prompt_template)

    def eval_fn(
        inputs=None,
        outputs=None,
        expectations=None,
        trace=None,
        **kwargs,
    ) -> Union[int, float, bool, str]:
        """
        This is the function that is called when the metric is evaluated (typically row-by-row).
        """
        kwargs.update({"inputs": inputs, "outputs": outputs, "expectations": expectations})
        grading_payloads = pd.DataFrame(kwargs).to_dict(orient="records")
        arg_strings = [prompt_template.format(**payload) for payload in grading_payloads]
        scores = _score_model_on_payloads(arg_strings, judge, {}, result_type, max_workers)
        return scores if len(scores) > 1 else scores[0]

    return scorer(name=name)(eval_fn)
