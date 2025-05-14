import os
from typing import Any, Optional, Union

import yaml

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

__mlflow_model_config__ = None


class ModelConfig:
    """
    ModelConfig used in code to read a YAML configuration file or a dictionary.

    Args:
        development_config: Path to the YAML configuration file or a dictionary containing the
                        configuration. If the configuration is not provided, an error is raised

    .. code-block:: python
        :caption: Example usage in model code

        from mlflow.models import ModelConfig

        # Load the configuration from a dictionary
        config = ModelConfig(development_config={"key1": "value1"})
        print(config.get("key1"))


    .. code-block:: yaml
        :caption: yaml file for model configuration

        key1: value1
        another_key:
            - value2
            - value3

    .. code-block:: python
        :caption: Example yaml usage in model code

        from mlflow.models import ModelConfig

        # Load the configuration from a file
        config = ModelConfig(development_config="config.yaml")
        print(config.get("key1"))


    When invoking the ModelConfig locally in a model file, development_config can be passed in
    which would be used as configuration for the model.


    .. code-block:: python
        :caption: Example to use ModelConfig when logging model as code: agent.py

        import mlflow
        from mlflow.models import ModelConfig

        config = ModelConfig(development_config={"key1": "value1"})


        class TestModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                return config.get("key1")


        mlflow.models.set_model(TestModel())


    But this development_config configuration file will be overridden when logging a model.
    When no model_config is passed in while logging the model, an error will be raised when
    trying to load the model using ModelConfig.
    Note: development_config is not used when logging the model.


    .. code-block:: python
        :caption: Example to use agent.py to log the model: deploy.py

        model_config = {"key1": "value2"}
        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                name="model", python_model="agent.py", model_config=model_config
            )

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

        # This will print "value2" as the model_config passed in while logging the model
        print(loaded_model.predict(None))
    """

    def __init__(self, *, development_config: Optional[Union[str, dict[str, Any]]] = None):
        config = globals().get("__mlflow_model_config__", None)
        # Here mlflow_model_config have 3 states:
        # 1. None, this means if the mlflow_model_config is None, use development_config if
        # available
        # 2. "", Empty string, this means the users explicitly didn't set the model config
        # while logging the model so if ModelConfig is used, it should throw an error
        # 3. A valid path, this means the users have set the model config while logging the
        # model so use that path
        if config is not None:
            self.config = config
        else:
            self.config = development_config

        if not self.config:
            raise FileNotFoundError(
                "Config file is not provided which is needed to load the model. "
                "Please provide a valid path."
            )

        if not isinstance(self.config, dict) and not os.path.isfile(self.config):
            raise FileNotFoundError(f"Config file '{self.config}' not found.")

    def _read_config(self):
        """Reads the YAML configuration file and returns its contents.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML content.

        Returns:
            dict or None: The content of the YAML file as a dictionary, or None if the
            config path is not set.
        """
        if isinstance(self.config, dict):
            return self.config

        with open(self.config) as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise MlflowException(
                    f"Error parsing YAML file: {e}", error_code=INVALID_PARAMETER_VALUE
                )

    def to_dict(self):
        """Returns the configuration as a dictionary."""
        return self._read_config()

    def get(self, key):
        """Gets the value of a top-level parameter in the configuration."""
        config_data = self._read_config()

        if config_data and key in config_data:
            return config_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in configuration: {config_data}.")


def _set_model_config(model_config):
    globals()["__mlflow_model_config__"] = model_config
