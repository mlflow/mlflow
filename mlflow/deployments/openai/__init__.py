import logging
import os

from mlflow.exceptions import MlflowException
from mlflow.openai import _OpenAIApiConfig, _OpenAIEnvVar
from mlflow.openai.utils import REQUEST_URL_CHAT
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)

from mlflow.deployments import BaseDeploymentClient


class OpenAIDeploymentClient(BaseDeploymentClient):
    """
    Client for interacting with OpenAI endpoints.

    Example:

    First, set up credentials for authentication:

    .. code-block:: bash

        export OPENAI_API_KEY=...

    .. seealso::

        See https://mlflow.org/docs/latest/python_api/openai/index.html for other authentication methods.

    Then, create a deployment client and use it to interact with OpenAI endpoints:

    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("openai")
        client.predict(endpoint="gpt-3.5-turbo", inputs={"prompt": "hello", "temperature": 0.1})
    """

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        if "OPENAI_API_KEY" not in os.environ:
            raise MlflowException(
                "OPENAI_API_KEY environment variable not set",
                error_code=INVALID_PARAMETER_VALUE,
            )

        from mlflow.openai.api_request_parallel_processor import process_api_requests
        from mlflow.openai.utils import _OAITokenHolder

        api_config = _get_api_config_without_openai_dep()
        api_token = _OAITokenHolder(api_config.api_type)

        if api_config.api_type in ("azure", "azure_ad", "azuread"):
            api_base = getattr(api_config, "api_base")
            api_version = getattr(api_config, "api_version")
            engine = getattr(api_config, "engine")
            deployment_id = getattr(api_config, "deployment_id")

            if engine:
                # Avoid using both parameters as they serve the same purpose
                # Invalid inputs:
                #   - Wrong engine + correct/wrong deployment_id
                #   - No engine + wrong deployment_id
                # Valid inputs:
                #   - Correct engine + correct/wrong deployment_id
                #   - No engine + correct deployment_id
                if deployment_id is not None:
                    _logger.warning(
                        "Both engine and deployment_id are set. "
                        "Using engine as it takes precedence."
                    )
                inputs = {"engine": engine, **inputs}
            elif deployment_id is None:
                raise MlflowException(
                    "Either engine or deployment_id must be set for Azure OpenAI API",
                )

            request_url = (
                f"{api_base}/openai/deployments/{deployment_id}"
                f"/chat/completions?api-version={api_version}"
            )
        else:
            inputs = {"model": endpoint, **inputs}
            request_url = REQUEST_URL_CHAT

        try:
            resp = process_api_requests(
                [inputs],
                request_url,
                api_token=api_token,
                throw_original_error=True,
                max_workers=1,
            )[0]
        except MlflowException as e:
            raise e
        except Exception as e:
            raise MlflowException(f"Error response from OpenAI:\n {e}")

        return resp

    def create_endpoint(self, name, config=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def update_endpoint(self, endpoint, config=None):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def delete_endpoint(self, endpoint):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def list_endpoints(self):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError

    def get_endpoint(self, endpoint):
        """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
        raise NotImplementedError


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass


def _get_api_config_without_openai_dep() -> _OpenAIApiConfig:
    """
    Gets the parameters and configuration of the OpenAI API connected to.
    """
    api_type = os.getenv(_OpenAIEnvVar.OPENAI_API_TYPE.value)
    api_version = os.getenv(_OpenAIEnvVar.OPENAI_API_VERSION.value)
    api_base = os.getenv(_OpenAIEnvVar.OPENAI_API_BASE.value, None)
    engine = os.getenv(_OpenAIEnvVar.OPENAI_ENGINE.value, None)
    deployment_id = os.getenv(_OpenAIEnvVar.OPENAI_DEPLOYMENT_NAME.value, None)
    if api_type in ("azure", "azure_ad", "azuread"):
        batch_size = 16
        max_tokens_per_minute = 60_000
    else:
        # The maximum batch size is 2048:
        # https://github.com/openai/openai-python/blob/b82a3f7e4c462a8a10fa445193301a3cefef9a4a/openai/embeddings_utils.py#L43
        # We use a smaller batch size to be safe.
        batch_size = 1024
        max_tokens_per_minute = 90_000
    return _OpenAIApiConfig(
        api_type=api_type,
        batch_size=batch_size,
        max_requests_per_minute=3_500,
        max_tokens_per_minute=max_tokens_per_minute,
        api_base=api_base,
        api_version=api_version,
        engine=engine,
        deployment_id=deployment_id,
    )
