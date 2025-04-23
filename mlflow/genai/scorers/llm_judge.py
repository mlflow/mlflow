from typing import Callable, Literal, Optional, Union

import pandas as pd
import pydantic

from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.evaluation import Assessment
from mlflow.genai.scorers import Scorer, scorer
from mlflow.metrics.genai.genai_metric import (
    _score_model_on_payloads,
)
from mlflow.metrics.genai.prompt_template import PromptTemplate

_PROMPT_SUFFIX = """

You must return the following fields in your response in two lines, one below the other:
score: Your score based on the rubric (could be a number or a string)
justification: Your reasoning for giving this score

Do not add additional new lines. Do not add any other fields."""


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
        true_values = ("true", "yes", "1", "t", "y")
        false_values = ("false", "no", "0", "f", "n")
        if lower_result in true_values:
            return True
        elif lower_result in false_values:
            return False
        else:
            raise ValueError(
                f"Could not convert LLM judge output to bool, judge output: '{lower_result}' "
                f"expected one of: {true_values + false_values}"
            )
    elif result_type == int:
        try:
            return int(str(raw_result).strip())
        except Exception as e:
            breakpoint()
            raise ValueError(
                f"Could not convert LLM judge output to int, judge output: '{raw_result}' "
                f"expected a valid integer"
            ) from e
    elif result_type == float:
        try:
            return float(str(raw_result).strip())
        except Exception as e:
            raise ValueError(
                f"Could not convert LLM judge output to float, judge output: '{raw_result}' "
                f"expected a valid float"
            ) from e
    elif result_type == str:
        return raw_result

    # TODO: add support for pydantic models

    raise ValueError(
        f"Unsupported result type in LLM judge: '{result_type!s}', expected "
        "one of: (bool, int, float, str)"
    )


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
            wrapped with curly braces e.g. {outputs} to fill-in any fields from
            the input dataset into prompts.
        judge: The judge model to score the assessment. Specify model provider and a model
            name in '<provider>:/<model>' format. Default is OpenAI GPT4o (tentative)
        result_type: A type of judge outputs.
        max_workers: Number of max parallelization to make LLM requests.
        aggregations: The list of options to aggregate the scores. Supported
            options are: min, max, mean, median, variance, p90.
            To use a custom aggregation, specify a function that takes a list of
            Assessment and outputs a Metric.

    Returns:
        A Scorer object representing the LLM judge.

    .. code-block:: python
        :test:
        :caption: Example for creating an llm judge scorer

        from mlflow.genai.scorers.llm_judge import llm_judge_scorer

        scorer = llm_judge_scorer(
            name="custom_correctness_judge",
            prompt_template=(
                "Does this answer the question correctly?\\nQuestion: {inputs}\\nAnswer: {outputs}"
            ),
            judge="openai:/gpt-4o",
            result_type=bool,
        )
    """
    # TODO: aggregations require updating databricks-agent eval
    # TODO: switch to structured generation API instead of optimistically casting to result_type

    prompt_template = PromptTemplate([prompt_template, _PROMPT_SUFFIX])

    def eval_fn(
        inputs=None,
        outputs=None,
        expectations=None,
        trace=None,
        **kwargs,
    ) -> Union[Assessment, list[Assessment]]:
        """
        This is the function that is called when the metric is evaluated (typically row-by-row).
        """
        kwargs.update({"inputs": inputs, "outputs": outputs, "expectations": expectations})
        prompt_args = pd.DataFrame(kwargs).to_dict(orient="records")
        prompts = [prompt_template.format(**_args) for _args in prompt_args]
        scores, justifications = _score_model_on_payloads(
            prompts, judge, {}, None, None, max_workers
        )
        assessments = []
        for (
            score,
            justification,
        ) in zip(scores, justifications):
            try:
                parsed_score = _cast_to_result_type(score, result_type)
                assessment = Assessment(
                    name=name,
                    source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
                    rationale=justification,
                    value=parsed_score,
                )
            except Exception as e:
                assessment = Assessment(
                    name=name,
                    source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
                    rationale=justification,
                    error_code="PARSE_ERROR",
                    error_message=str(e),
                )

            assessments.append(assessment)

        return assessments if len(assessments) > 1 else assessment

    return scorer(name=name)(eval_fn)
