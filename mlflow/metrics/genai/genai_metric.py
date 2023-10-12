import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample, MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.utils import _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string

if TYPE_CHECKING:
    import pandas as pd

_logger = logging.getLogger(__name__)


def _format_args_string(grading_context_columns: Optional[List[str]], eval_values, indx) -> str:
    args_dict = {}
    for arg in grading_context_columns:
        if arg in eval_values:
            args_dict[arg] = eval_values[arg][indx]
        else:
            raise MlflowException(
                f"{arg} does not exist in the eval function {list(eval_values.keys())}."
            )

    return (
        ""
        if args_dict is None
        else "\n".join(f"Provided {arg}: {arg_value}" for arg, arg_value in args_dict.items())
    )


# Function to extract Score and Justification
def _extract_score_and_justification(output):
    if (
        isinstance(output, dict)
        and "candidates" in output
        and isinstance(output["candidates"], list)
        and output["candidates"]
    ):
        text = output["candidates"][0]["text"]

    if text:
        # Attempt to parse JSON
        try:
            data = json.loads(text)
            score = int(data.get("Score"))
            justification = data.get("Justification")
        except json.JSONDecodeError:
            # If parsing fails, use regex
            match = re.search(r"Score: (\d+),?\s*Justification: (.+)", text)
            if match:
                score = int(match.group(1))
                justification = match.group(2)
            else:
                score = None
                justification = None

        if not isinstance(score, (int, float)) or not isinstance(justification, str):
            return None, None

        return score, justification

    return None, None


@experimental
def make_genai_metric(
    name: str,
    definition: str,
    grading_prompt: str,
    examples: Optional[List[EvaluationExample]] = None,
    version: Optional[str] = _get_latest_metric_version(),
    model: Optional[str] = "openai:/gpt-3.5-turbo-16k",
    grading_context_columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    aggregations: Optional[List[str]] = None,
    greater_is_better: bool = True,
    max_workers: int = 10,
    judge_request_timeout: int = 15,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLM using LLM as a judge in MLflow.

    :param name: Name of the metric.
    :param definition: Definition of the metric.
    :param grading_prompt: Grading criteria of the metric.
    :param examples: (Optional) Examples of the metric.
    :param version: (Optional) Version of the metric. Currently supported versions are: v1.
    :param model: (Optional) Model uri of the metric.
    :param grading_context_columns: (Optional) grading_context_columns required to compute
        the metric. These grading_context_columns are used by the LLM as a judge as additional
        information to compute the metric. The columns are extracted from the input dataset or
        output predictions based on col_mapping in evaluator_config.
    :param parameters: (Optional) Parameters for the llm used to compute the metric.
    :param aggregations: (Optional) The list of options to aggregate the scores. Currently supported
        options are: min, max, mean, median, variance, p90.
    :param greater_is_better: (Optional) Whether the metric is better when it is greater.
    :param max_workers: (Optional) The maximum number of workers to use for judge scoring.
        Defaults to 10 workers.
    :param judge_request_timeout: (Optional) The timeout in seconds for each judge scoring request.
        Defaults to 15 seconds.

    :return: A metric object.

    .. testcode:: python
        :caption: Example for creating a genai metric

        from mlflow.metrics import EvaluationExample, make_genai_metric

        example = EvaluationExample(
            input="What is MLflow?",
            output=(
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle."
            ),
            score=4,
            justification=(
                "The definition effectively explains what MLflow is "
                "its purpose, and its developer. It could be more concise for a 5-score.",
            ),
            grading_context={
                "ground_truth": (
                    "MLflow is an open-source platform for managing "
                    "the end-to-end machine learning (ML) lifecycle. It was developed by "
                    "Databricks, a company that specializes in big data and machine learning "
                    "solutions. MLflow is designed to address the challenges that data "
                    "scientists and machine learning engineers face when developing, training, "
                    "and deploying machine learning models."
                )
            },
        )

        metric = make_genai_metric(
            name="correctness",
            definition=(
                "Correctness refers to how well the generated output matches "
                "or aligns with the reference or ground truth text that is considered "
                "accurate and appropriate for the given input. The ground truth serves as "
                "a benchmark against which the provided output is compared to determine the "
                "level of accuracy and fidelity."
            ),
            grading_prompt=(
                "Correctness: If the answer correctly answer the question, below "
                "are the details for different scores: "
                "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
                "the question or is completely contrary to the correct answer. "
                "- Score 1: the answer provides some relevance to the question and answer "
                "one aspect of the question correctly. "
                "- Score 2: the answer mostly answer the question but is missing or hallucinating "
                "on one critical aspect. "
                "- Score 4: the answer correctly answer the question and not missing any "
                "major aspect"
            ),
            examples=[example],
            version="v1",
            model="gateway:/gpt4",
            grading_context_columns=["ground_truth"],
            parameters={"temperature": 1.0},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True,
        )
    """

    if aggregations is None:
        aggregations = ["mean", "variance", "p90"]

    def eval_fn(
        predictions: "pd.Series",
        metrics: Dict[str, MetricValue],
        inputs: "pd.Series",
        *args,
    ) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        """

        eval_values = dict(zip(grading_context_columns, args))
        class_name = f"mlflow.metrics.genai.prompts.{version}.EvaluationModel"
        try:
            evaluation_model_class_module = _get_class_from_string(class_name)
        except ModuleNotFoundError:
            raise MlflowException(
                f"Failed to find evaluation model for version {version}."
                f"Please check the correctness of the version",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        except Exception as e:
            raise MlflowException(
                f"Failed to construct evaluation model {version}. Error: {e!r}",
                error_code=INTERNAL_ERROR,
            ) from None

        evaluation_context = evaluation_model_class_module(
            name,
            definition,
            grading_prompt,
            examples,
            model,
            *(parameters,) if parameters is not None else (),
        ).to_dict()

        outputs = predictions.to_list()
        inputs = inputs.to_list()
        eval_model = evaluation_context["model"]
        eval_parameters = evaluation_context["parameters"]

        # TODO: Save the metric definition in a yaml file for model monitoring

        if not isinstance(eval_model, str):
            raise MlflowException(
                message="The model argument must be a string URI referring to an openai model "
                "(openai:/gpt-3.5-turbo) or  gateway (gateway:/my-route), "
                f"passed {eval_model} instead",
                error_code=INVALID_PARAMETER_VALUE,
            )

        def score_model_on_one_payload(
            indx,
            input,
            output,
            grading_context_columns,
            eval_values,
            evaluation_context,
            eval_parameters,
            eval_model,
        ):
            arg_string = _format_args_string(grading_context_columns, eval_values, indx)
            payload = {
                "prompt": evaluation_context["eval_prompt"].format(
                    input=input, output=output, grading_context_columns=arg_string
                ),
                **eval_parameters,
            }
            try:
                raw_result = model_utils.score_model_on_payload(eval_model, payload)
                return _extract_score_and_justification(raw_result)
            except Exception as e:
                _logger.info(f"Failed to score model on payload. Error: {e!r}")
                return None, None

        scores = [None] * len(inputs)
        justifications = [None] * len(inputs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    score_model_on_one_payload,
                    indx,
                    input,
                    output,
                    grading_context_columns,
                    eval_values,
                    evaluation_context,
                    eval_parameters,
                    eval_model,
                ): indx
                for indx, (input, output) in enumerate(zip(inputs, outputs))
            }

            for future in as_completed(futures, timeout=judge_request_timeout):
                indx = futures[future]
                score, justification = future.result()
                scores[indx] = score
                justifications[indx] = justification

        # loop over the aggregations and compute the aggregate results on the scores
        def aggregate_function(aggregate_option, scores):
            import numpy as np

            options = {
                "min": np.min,
                "max": np.max,
                "mean": np.mean,
                "median": np.median,
                "variance": np.var,
                "p90": lambda x: np.percentile(x, 90) if x else None,
            }

            if aggregate_option not in options:
                raise MlflowException(
                    message=f"Invalid aggregate option {aggregate_option}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            return options[aggregate_option](scores)

        scores_for_aggregation = [score for score in scores if score is not None]

        aggregate_results = (
            {option: aggregate_function(option, scores_for_aggregation) for option in aggregations}
            if aggregations is not None
            else {}
        )

        return MetricValue(scores, justifications, aggregate_results)

    signature_parameters = [
        Parameter("predictions", Parameter.POSITIONAL_OR_KEYWORD, annotation="pd.Series"),
        Parameter("metrics", Parameter.POSITIONAL_OR_KEYWORD, annotation=Dict[str, MetricValue]),
        Parameter("inputs", Parameter.POSITIONAL_OR_KEYWORD, annotation="pd.Series"),
    ]

    # Add grading_context_columns to signature list
    for var in grading_context_columns:
        signature_parameters.append(Parameter(var, Parameter.POSITIONAL_OR_KEYWORD))

    eval_fn.__signature__ = Signature(signature_parameters)

    return make_metric(
        eval_fn=eval_fn, greater_is_better=greater_is_better, name=name, version=version
    )
