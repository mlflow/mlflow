import os

from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)


def get_env(variable_name):
    return os.environ.get(variable_name)


def unset_variable(variable_name):
    if variable_name in os.environ:
        del os.environ[variable_name]


_env_parser_and_default_value_map = {
    MLFLOW_HTTP_REQUEST_MAX_RETRIES: (int, 5),
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR: (int, 2),
    MLFLOW_HTTP_REQUEST_TIMEOUT: (int, 120),
}


def _get_env_config_value_or_default(name):
    assert name in _env_parser_and_default_value_map, f"Invalid environment config name {name}."
    parser, default_value = _env_parser_and_default_value_map[name]
    if name in os.environ:
        value = os.environ[name]
        try:
            return parser(value)
        except Exception as e:
            raise ValueError(
                f"Parse environment config {name}'s value '{value}' failed. (error: {repr(e)})"
            )
    else:
        return default_value
