"""
The ``mlflow.openai`` module provides an API for logging and loading OpenAI models.

Credential management for OpenAI on Databricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    Specifying secrets for model serving with ``MLFLOW_OPENAI_SECRET_SCOPE`` is deprecated.
    Use `secrets-based environment variables <https://docs.databricks.com/en/machine-learning/model-serving/store-env-variable-model-serving.html>`_
    instead.

When this flavor logs a model on Databricks, it saves a YAML file with the following contents as
``openai.yaml`` if the ``MLFLOW_OPENAI_SECRET_SCOPE`` environment variable is set.

.. code-block:: yaml

    OPENAI_API_BASE: {scope}:openai_api_base
    OPENAI_API_KEY: {scope}:openai_api_key
    OPENAI_API_KEY_PATH: {scope}:openai_api_key_path
    OPENAI_API_TYPE: {scope}:openai_api_type
    OPENAI_ORGANIZATION: {scope}:openai_organization

- ``{scope}`` is the value of the ``MLFLOW_OPENAI_SECRET_SCOPE`` environment variable.
- The keys are the environment variables that the ``openai-python`` package uses to
  configure the API client.
- The values are the references to the secrets that store the values of the environment
  variables.

When the logged model is served on Databricks, each secret will be resolved and set as the
corresponding environment variable. See https://docs.databricks.com/security/secrets/index.html
for how to set up secrets on Databricks.
"""

from mlflow.openai.autolog import autolog
from mlflow.openai.constant import FLAVOR_NAME
from mlflow.version import IS_TRACING_SDK_ONLY

__all__ = ["autolog", "FLAVOR_NAME"]


# Import model logging APIs only if mlflow skinny or full package is installed,
# i.e., skip if only mlflow-tracing package is installed.
if not IS_TRACING_SDK_ONLY:
    from mlflow.openai.model import (
        _load_pyfunc,
        load_model,
        log_model,
        save_model,
    )

    __all__ += [
        "load_model",
        "log_model",
        "save_model",
        "_load_pyfunc",
    ]
