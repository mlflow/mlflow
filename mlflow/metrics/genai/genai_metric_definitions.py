from typing import List, Optional

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.genai.make_genai_metric import make_genai_metric
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.class_utils import _get_class_from_string

DEFAULT_MODEL = "openai:/gpt4"
LATEST_VERSION = "v1"


def correctness(
    model: Optional[str] = DEFAULT_MODEL,
    version: Optional[str] = LATEST_VERSION,
    examples: Optional[List[EvaluationExample]] = [],
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the correctness of an LLM using the
    model provided. Correctness will be assessed by whether answers align exactly with ground truth
    for questions requiring precise responses, and by the similarity in meaning and description to
    the ground truth for questions allowing varied descriptions.

    An MlflowException will be raised if the specified version for this metric does not exist.

    :param model: The model that will be used to evaluate this metric
    :param version: The version of the correctness metric to use. Defaults to the latest version.
    :param examples: Provide a list of examples to help the judge model evaluate the correctness
    :return: A metric object
    """
    class_name = f"mlflow.metrics.utils.prompts.{version}.CorrectnessMetric"
    try:
        correctness_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find correctness metric for version {version}." f"Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct correctness metric {version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    return make_genai_metric(
        name="correctness",
        definition=correctness_class_module.definition,
        grading_prompt=correctness_class_module.grading_prompt,
        examples=examples,
        version=version,
        model=model,
        variables=correctness_class_module.variables,
        parameters=correctness_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )
