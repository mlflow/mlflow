from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample, MetricValue
from mlflow.metrics.utils import model_utils
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.class_utils import _get_class_from_string

if TYPE_CHECKING:
    import pandas as pd
    import pyspark


def _format_variable_string(variables: Dict[str, Any], eval_df, indx) -> str:
    variables_dict = {}
    for variable in variables:
        if variable in eval_df.columns:
            variables_dict[variable] = eval_df[variable].tolist()[indx]
        else:
            raise MlflowException(
                f"{variable} does not exist in the Eval DataFrame {eval_df.columns}."
            )

    variable_string = (
        ""
        if variables_dict is None
        else "\n".join(
            f"Provided {variable}: {variable_value}"
            for variable, variable_value in variables_dict.items()
        )
    )

    return variable_string


def make_genai_metric(
    name: str,
    definition: str,
    grading_prompt: str,
    examples: Optional[List[EvaluationExample]] = None,
    version: Optional[str] = "v1",
    model: Optional[str] = "openai:/gpt4",
    variables: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    aggregations: Optional[List[str]] = None,
    greater_is_better: bool = True,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLM using LLM as a judge in MLflow.

    :param name: Name of the metric.
    :param definition: Definition of the metric.
    :param grading_prompt: Grading criteria of the metric.
    :param examples: (Optional) Examples of the metric.
    :param version: (Optional) Version of the metric. Currently supported versions are: v1.
    :param model: (Optional) Model uri of the metric.
    :param variables: (Optional) Variables required to compute the metric.
    :param parameters: (Optional) Parameters for the llm used to compute the metric.
    :param aggregations: (Optional) The list of options to aggregate the scores. Currently supported
        options are: min, max, mean, median, variance, p90.
    :param greater_is_better: (Optional) Whether the metric is better when it is greater.

    :return: A metric object.

    .. test-code-block:: python
        :caption: Example for creating a genai metric

        from mlflow.metrics.base import EvaluationExample, make_genai_metric

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
            variables={
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
            variables=["ground_truth"],
            parameters={"temperature": 1.0},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True,
        )
    """

    def eval_fn(eval_df: Union["pd.DataFrame", "pyspark.sql.DataFrame"]) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        """

        class_name = f"mlflow.metrics.utils.prompts.{version}.EvaluationModel"
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

        outputs = eval_df["prediction"].tolist()
        inputs = eval_df["input"].tolist()
        eval_model = evaluation_context["model"]
        eval_parameters = evaluation_context["parameters"]

        # TODO: Save the metric definition in a yaml file for model monitoring

        payload = []
        for indx, (input, output) in enumerate(zip(inputs, outputs)):
            variable_string = _format_variable_string(variables, eval_df, indx)
            payload.append(
                {
                    "prompt": evaluation_context["eval_prompt"].format(
                        input=input, output=output, variables=variable_string
                    ),
                    **eval_parameters,
                },
            )

        eval_result = None
        if isinstance(eval_model, str):
            # TODO: Add batch processing for messages here
            eval_result = model_utils.score_model_on_payload(eval_model, payload)
        else:
            raise MlflowException(
                message="The model argument must be a string URI referring to an openai model "
                "(openai:/gpt-3.5-turbo) or  gateway (gateway:/my-route), "
                f"passed {eval_model} instead",
                error_code=INVALID_PARAMETER_VALUE,
            )

        scores = eval_result["Score"]
        justification = eval_result["Justification"]

        # loop over the aggregations and compute the aggregate results on the scores
        def aggregate_function(aggregate_option, scores):
            import numpy as np

            options = {
                "min": np.min,
                "max": np.max,
                "mean": np.mean,
                "median": np.median,
                "variance": np.var,
                "p90": lambda x: np.percentile(x, 90),
            }

            if aggregate_option not in options:
                raise MlflowException(
                    message=f"Invalid aggregate option {aggregate_option}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            return options[aggregate_option](scores)

        aggregate_results = {option: aggregate_function(option, scores) for option in aggregations}

        return MetricValue(scores.tolist(), justification.tolist(), aggregate_results)

    return make_metric(
        eval_fn=eval_fn, greater_is_better=greater_is_better, name=name, version=version
    )
