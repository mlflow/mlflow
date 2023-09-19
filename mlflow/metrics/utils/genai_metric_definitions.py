from mlflow.exceptions import MlflowException
from mlflow.metrics.utils import make_genai_metric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.class_utils import _get_class_from_string

DEFAULT_MODEL = "openai:/gpt4"
LATEST_VERSION = "v1"


def correctness(model=DEFAULT_MODEL, version=LATEST_VERSION, examples=[]):
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

    return make_genai_metric(
        name="correctness",
        definition=evaluation_model_class_module.correctness_definition,
        grading_prompt=evaluation_model_class_module.correctness_grading_prompt,
        examples=examples,
        version=version,
        model=model,
        variables=evaluation_model_class_module.variables,
        parameters=evaluation_model_class_module.correctness_parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )
