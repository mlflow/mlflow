import logging
from typing import Any, Dict, Union

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.scoring_server import _parse_json_data

_logger = logging.getLogger(__name__)


# NB: this function should always be kept in sync with the serving
# process in scoring_server invocations.
def validate_serving_input(model_uri: str, serving_input: Union[str, Dict[str, Any]]):
    """
    Helper function to validate the model can be served and provided input is valid
    prior to serving the model.

    Args:
        model_uri: URI of the model to be served.
        serving_input: Input data to be validated. Should be a dictionary or a JSON string.
    """

    try:
        # sklearn model might not have python_function flavor if it
        # doesn't define a predict function.
        # we should not fail if pyfunc model does not exist as the
        # model can not be served anyways
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise MlflowException(f"Failed to load model for serving validation: {e}")
    parsed_input = _parse_json_data(
        serving_input,
        pyfunc_model.metadata,
        pyfunc_model.metadata.get_input_schema(),
    )
    return pyfunc_model.predict(parsed_input.data, params=parsed_input.params)
