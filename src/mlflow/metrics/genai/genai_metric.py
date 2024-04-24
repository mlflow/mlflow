import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string

if TYPE_CHECKING:
    import pandas as pd

_logger = logging.getLogger(__name__)


def _format_args_string(grading_context_columns: Optional[List[str]], eval_values, indx) -> str:
    import pandas as pd

    args_dict = {}
    for arg in grading_context_columns:
        if arg in eval_values:
            args_dict[arg] = (
                eval_values[arg].iloc[indx]
                if isinstance(eval_values[arg], pd.Series)
                else eval_values[arg][indx]
            )
        else:
            raise MlflowException(
                f"{arg} does not exist in the eval function {list(eval_values.keys())}."
            )

    return (
        ""
        if args_dict is None or len(args_dict) == 0
        else (
            "Additional information used by the model:\n"
            + "\n".join(
                [f"key: {arg}\nvalue:\n{arg_value}" for arg, arg_value in args_dict.items()]
            )
        )
    )


# Function to extract Score and Justification
def _extract_score_and_justification(text):
    if text:
        text = re.sub(r"score", "score", text, flags=re.IGNORECASE)
        text = re.sub(r"justification", "justification", text, flags=re.IGNORECASE)
        # Attempt to parse JSON
        try:
            data = json.loads(text)
            score = int(data.get("score"))
            justification = data.get("justification")
        except json.JSONDecodeError:
            # If parsing fails, use regex
            if (match := re.search(r"score: (\d+),?\s*justification: (.+)", text)) or (
                match := re.search(r"\s*score:\s*(\d+)\s*justification:\s*(.+)", text, re.DOTALL)
            ):
                score = int(match.group(1))
                justification = match.group(2)
            else:
                score = None
                justification = f"Failed to extract score and justification. Raw output: {text}"

        if not isinstance(score, (int, float)) or not isinstance(justification, str):
            return None, f"Failed to extract score and justification. Raw output: {text}"

        return score, justification

    return None, None


@experimental
def make_genai_metric(
    name: str,
    definition: str,
    grading_prompt: str,
    examples: Optional[List[EvaluationExample]] = None,
    version: Optional[str] = _get_latest_metric_version(),
    model: Optional[str] = _get_default_model(),
    grading_context_columns: Optional[Union[str, List[str]]] = [],  # noqa: B006
    include_input: bool = True,
    parameters: Optional[Dict[str, Any]] = None,
    aggregations: Optional[List[str]] = ["mean", "variance", "p90"],  # noqa: B006
    greater_is_better: bool = True,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLM using LLM as a judge in MLflow. The full grading
    prompt is stored in the metric_details field of the ``EvaluationMetric`` object.

    Args:
        name: Name of the metric.
        definition: Definition of the metric.
        grading_prompt: Grading criteria of the metric.
        examples: (Optional) Examples of the metric.
        version: (Optional) Version of the metric. Currently supported versions are: v1.
        model: (Optional) Model uri of an openai, gateway, or deployments judge model in the
            format of "openai:/gpt-4", "gateway:/my-route",
            "endpoints:/databricks-llama-2-70b-chat".  Defaults to "openai:/gpt-4". If using
            Azure OpenAI, the ``OPENAI_DEPLOYMENT_NAME`` environment variable will take precedence.
            Your use of a third party LLM service (e.g., OpenAI) for evaluation may be subject to
            and governed by the LLM service's terms of use.
        grading_context_columns: (Optional) The name of the grading context column, or a list of
            grading context column names, required to compute the metric. The
            ``grading_context_columns`` are used by the LLM as a judge as additional information to
            compute the metric. The columns are extracted from the input dataset or output
            predictions based on ``col_mapping`` in the ``evaluator_config`` passed to
            :py:func:`mlflow.evaluate()`. They can also be the name of other evaluated metrics.
        include_input: (Optional) Whether to include the input
            when computing the metric.
        parameters: (Optional) Parameters for the LLM used to compute the metric. By default, we
            set the temperature to 0.0, max_tokens to 200, and top_p to 1.0. We recommend
            setting the temperature to 0.0 for the LLM used as a judge to ensure consistent results.
        aggregations: (Optional) The list of options to aggregate the scores. Currently supported
            options are: min, max, mean, median, variance, p90.
        greater_is_better: (Optional) Whether the metric is better when it is greater.
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object.

    .. code-block:: python
        :test:
        :caption: Example for creating a genai metric

        from mlflow.metrics.genai import EvaluationExample, make_genai_metric

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
                "targets": (
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
            name="answer_correctness",
            definition=(
                "Answer correctness is evaluated on the accuracy of the provided output based on "
                "the provided targets, which is the ground truth. Scores can be assigned based on "
                "the degree of semantic similarity and factual correctness of the provided output "
                "to the provided targets, where a higher score indicates higher degree of accuracy."
            ),
            grading_prompt=(
                "Answer correctness: Below are the details for different scores:"
                "- Score 1: The output is completely incorrect. It is completely different from "
                "or contradicts the provided targets."
                "- Score 2: The output demonstrates some degree of semantic similarity and "
                "includes partially correct information. However, the output still has significant "
                "discrepancies with the provided targets or inaccuracies."
                "- Score 3: The output addresses a couple of aspects of the input accurately, "
                "aligning with the provided targets. However, there are still omissions or minor "
                "inaccuracies."
                "- Score 4: The output is mostly correct. It provides mostly accurate information, "
                "but there may be one or more minor omissions or inaccuracies."
                "- Score 5: The output is correct. It demonstrates a high degree of accuracy and "
                "semantic similarity to the targets."
            ),
            examples=[example],
            version="v1",
            model="openai:/gpt-4",
            grading_context_columns=["targets"],
            parameters={"temperature": 0.0},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True,
        )
    """
    if not isinstance(grading_context_columns, list):
        grading_context_columns = [grading_context_columns]

    def process_example(example):
        if example.grading_context is None and len(grading_context_columns) == 0:
            grading_context = {}
        elif isinstance(example.grading_context, dict):
            grading_context = example.grading_context
        else:
            # The grading context is string-like. Assume that it corresponds to the first
            # grading context column and update the example accordingly
            grading_context = {grading_context_columns[0]: example.grading_context}
            example.grading_context = grading_context

        if set(grading_context.keys()) != set(grading_context_columns):
            raise MlflowException.invalid_parameter_value(
                f"Example grading context does not contain required columns.\n"
                f" Example grading context columns: {list(grading_context.keys())}\n"
                f" Required grading context columns: {grading_context_columns}\n"
            )

        if not include_input:
            return EvaluationExample(
                output=example.output,
                score=example.score,
                justification=example.justification,
                grading_context=example.grading_context,
            )

        return example

    if examples is not None:
        examples = [process_example(example) for example in examples]

    class_name = f"mlflow.metrics.genai.prompts.{version}.EvaluationModel"
    try:
        evaluation_model_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find evaluation model for version {version}."
            f" Please check the correctness of the version",
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

        outputs = predictions.to_list()
        inputs = inputs.to_list()
        eval_model = evaluation_context["model"]
        eval_parameters = evaluation_context["parameters"]

        # TODO: Save the metric definition in a yaml file for model monitoring

        if not isinstance(eval_model, str):
            raise MlflowException(
                message="The model argument must be a string URI referring to an openai model "
                "(openai:/gpt-3.5-turbo) or an MLflow Deployments endpoint "
                f"(endpoints:/my-endpoint), passed {eval_model} instead",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # generate grading payloads
        grading_payloads = []
        for indx, (input, output) in enumerate(zip(inputs, outputs)):
            try:
                arg_string = _format_args_string(grading_context_columns, eval_values, indx)
            except Exception as e:
                raise MlflowException(
                    f"Values for grading_context_columns are malformed and cannot be "
                    f"formatted into a prompt for metric '{name}'.\n"
                    f"Required columns: {grading_context_columns}\n"
                    f"Values: {eval_values}\n"
                    f"Error: {e!r}\n"
                    f"Please check the following: \n"
                    "- predictions and targets (if required) are provided correctly\n"
                    "- grading_context_columns are mapped correctly using the evaluator_config "
                    "parameter\n"
                    "- input and output data are formatted correctly."
                )
            grading_payloads.append(
                evaluation_context["eval_prompt"].format(
                    input=(input if include_input else None),
                    output=output,
                    grading_context_columns=arg_string,
                )
            )

        def score_model_on_one_payload(
            payload,
            eval_model,
        ):
            try:
                raw_result = model_utils.score_model_on_payload(
                    eval_model, payload, eval_parameters
                )
                return _extract_score_and_justification(raw_result)
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

        scores = [None] * len(inputs)
        justifications = [None] * len(inputs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    score_model_on_one_payload,
                    payload,
                    eval_model,
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
        eval_fn=eval_fn,
        greater_is_better=greater_is_better,
        name=name,
        version=version,
        metric_details=evaluation_context["eval_prompt"].__str__(),
    )
