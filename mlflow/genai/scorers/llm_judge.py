from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
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
    # TODO: cast the output of the LLM to a pydantic basemodel or a primitive type
    pass


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
) -> list[Union[type, pydantic.BaseModel]]:
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
    result_type: Union[type, pydantic.BaseModel] = bool,
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
    allowed_variables = prompt_template.variables

    def eval_fn(
        *args,
        **kwargs,
    ) -> Union[
        type, pydantic.BaseModel
    ]:  # TODO: update this to be exactly result_type? or leave it
        """
        This is the function that is called when the metric is evaluated.
        """
        if missing_variables := allowed_variables - set(kwargs.keys()):
            raise MlflowException(
                message=f"Missing variable inputs to eval_fn: {missing_variables}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        kwargs = {k: [v] if np.isscalar(v) else v for k, v in kwargs.items()}
        grading_payloads = pd.DataFrame(kwargs).to_dict(orient="records")
        arg_strings = [prompt_template.format(**payload) for payload in grading_payloads]
        scores, _ = _score_model_on_payloads(arg_strings, judge, {}, result_type, max_workers)

        return scores

    if allowed_variables:
        eval_fn.__signature__ = Signature(
            parameters=[
                Parameter(name=var, kind=Parameter.KEYWORD_ONLY) for var in allowed_variables
            ]
        )

    return scorer(name=name)(eval_fn)
