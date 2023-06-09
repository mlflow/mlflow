import os
from typing import List

from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.gateway.handlers import _load_route_config, RouteConfig


def _validate_config(config_path: str) -> List[RouteConfig]:
    if not os.path.exists(config_path):
        raise MlflowException.invalid_parameter_value(f"{config_path} does not exist")

    try:
        return _load_route_config(config_path)
    except ValidationError as e:
        raise MlflowException.invalid_parameter_value(f"Invalid gateway configuration: {e}") from e
