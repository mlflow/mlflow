import os
from typing import List

import yaml
from pydantic import BaseModel, ValidationError, Extra

from mlflow.exceptions import MlflowException


class Config(BaseModel, extra=Extra.forbid):
    routes: List[str]


def _validate_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise MlflowException.invalid_parameter_value(f"{config_path} does not exist")

    with open(config_path) as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"{config_path} is not a valid YAML file"
            ) from e
    try:
        return Config(**cfg)
    except ValidationError as e:
        raise MlflowException.invalid_parameter_value(f"Invalid gateway configuration: {e}") from e
