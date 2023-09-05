from typing import Any, Dict, List, Union

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample, MetricValue
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.pyfunc import PyFuncModel, _load_model_or_server
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.class_utils import _get_class_from_string


def make_genai_metric(
    name: str,
    version: str,
    definition: str,
    grading_prompt: str,
    examples: List[EvaluationExample] = None,
    model: str = "openai:/gpt4",
    variables: List[str] = None,
    parameters: Dict[str, Any] = None,
    greater_is_better=True,
    aggregate_options: List[str] = None,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLm using LLM as a judge in MLflow.

    :param name: Name of the metric.
    :param version: Version of the metric. Currently supported versions are: v1.
    :param model: Model uri of the metric.
    :param variables: Variables required to compute the metric.
    :param definition: Definition of the metric.
    :param grading_prompt: Grading criteria of the metric.
    :param examples: Examples of the metric.
    :param parameters: Parameters for the llm used to compute the metric.
    :param greater_is_better: Whether the metric is better when it is greater.
    :param aggregate_options: The list of options to aggregate the scores. Currently supported
        options are: min, max, mean, median, variance, p90.

    :return: A metric object.

    .. test-code-block:: python
        :caption: Example for creating a genai metric

        from mlflow.metrics.base import EvaluationExample, make_genai_metric

        example = EvaluationExample(
            input="What is MLflow?",
            output="MLflow is an open-source platform for managing machine "
            "learning workflows, including experiment tracking, model packaging, "
            "versioning, and deployment, simplifying the ML lifecycle.",
            score=4,
            justification="The definition effectively explains what MLflow is "
            "its purpose, and its developer. It could be more concise for a 5-score.",
            variables={
                "ground_truth": "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
                "a company that specializes in big data and machine learning solutions. MLflow is "
                "designed to address the challenges that data scientists and machine learning "
                "engineers face when developing, training, and deploying machine learning models."
            },
        )

        metric = make_genai_metric(
            name="correctness",
            version="v1",
            definition="Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity.",
            grading_prompt="Correctness: If the answer correctly answer the question, below are the "
            "details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn’t mention anything about the question "
            "or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer one aspect "
            "of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
            "critical aspect. "
            "- Score 4: the answer correctly answer the question and not missing any major aspect",
            examples=[example],
            model="gateway:/gpt4",
            variables=["ground_truth"],
            parameters={"temperature": 1.0},
            greater_is_better=True,
            aggregate_options=["mean", "variance", "p90"],
        )
    """
    import pandas as pd
    import pyspark

    def eval_fn(eval_df: Union[pd.DataFrame, pyspark.sql.DataFrame]) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        """

        class_name = f"mlflow.metrics.utils.prompts.{version}.EvaluationModel"
        try:
            evaluation_model_class_module = _get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find evaluation model for version {version}."
                    f"Please check the correctness of the version",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct evaluation model {version}. Error: {e!r}",
                    error_code=INTERNAL_ERROR,
                ) from None

        variables_dict = {}
        for variable in variables:
            if variable in eval_df.columns:
                variable_value = eval_df[variable]
                variables_dict[variable] = variable_value
            else:
                raise MlflowException(
                    f"{variable} does not exist in the Eval DataFrame {eval_df.columns}."
                )

        variable_string = (
            ""
            if variables_dict is None
            else "\n".join(
                [
                    f"Provided {variable}: {variable_value}"
                    for variable, variable_value in variables_dict.items()
                ]
            )
        )

        evaluation_context = evaluation_model_class_module(
            name,
            definition,
            grading_prompt,
            examples,
            model,
            parameters,
        ).to_dict()

        outputs = eval_df["prediction"]
        inputs = eval_df["input"]
        eval_model = evaluation_context["model"]
        eval_parameters = evaluation_context["parameters"]

        # TODO: Save the metric definition in a yaml file for model monitoring

        messages = []
        for indx, _ in inputs.items():
            messages.append(
                [
                    evaluation_context["eval_prompt"].partial_fill(
                        input=inputs[indx], output=outputs[indx], variables=variable_string
                    ),
                ],
            )

        # TODO: load the model for openai and gateway URI
        if isinstance(eval_model, str):
            eval_model = _load_model_or_server(eval_model, _EnvManager.LOCAL)
        elif isinstance(eval_model, PyFuncModel):
            pass
        else:
            raise MlflowException(
                message="The model argument must be a string URI referring to an MLflow model or "
                "an instance of `mlflow.pyfunc.PyFuncModel`.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # TODO: Add batch processing for messages here
        eval_result = eval_model.predict(messages, eval_parameters)
        scores = eval_result["Score"]
        justification = eval_result["Justification"]

        # loop over the aggregate_options and compute the aggregate results on the scores

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

        aggregate_results = {
            option: aggregate_function(option, scores) for option in aggregate_options
        }

        return MetricValue(scores.tolist(), justification.tolist(), aggregate_results)

    return make_metric(
        eval_fn=eval_fn, greater_is_better=greater_is_better, name=name, version=version
    )
