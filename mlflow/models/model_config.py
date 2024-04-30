import os
from typing import Any, Dict, Optional, Union

import yaml

__mlflow_model_config__ = None


# TODO: Let ModelConfig take in a dictionary instead of a file path
class ModelConfig:
    """
    ModelConfig used in code to read a YAML configuration file, and this configuration file can be
    overridden when logging a model.
    """

    def __init__(self, *, development_config: Optional[Union[str, Dict[str, Any]]] = None):
        config = globals().get("__mlflow_model_config__", None)
        # backwards compatibility
        rag_config = globals().get("__databricks_rag_config_path__", None)
        self.config = config or rag_config or development_config

        if not self.config:
            raise FileNotFoundError("Config file is None. Please provide a valid path.")

        if not isinstance(development_config, dict) and not os.path.isfile(self.config):
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
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    def get(self, key):
        """Gets the value of a top-level parameter in the configuration."""
        config_data = self._read_config()

        if config_data and key in config_data:
            return config_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in configuration: {config_data}.")


def _set_model_config(model_config):
    globals()["__mlflow_model_config__"] = model_config
