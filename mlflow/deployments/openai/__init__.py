import logging
import os

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE, UNAUTHENTICATED

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
        payload = inputs
        if "OPENAI_API_KEY" not in os.environ:
            raise MlflowException(
                "OPENAI_API_KEY environment variable not set",
                error_code=INVALID_PARAMETER_VALUE,
            )

        import openai
        import openai.error

        from mlflow.openai import _get_api_config
        from mlflow.openai.api_request_parallel_processor import process_api_requests
        from mlflow.openai.utils import _OAITokenHolder

        api_config = _get_api_config()
        api_token = _OAITokenHolder(api_config.api_type)
        envs = {
            x: getattr(api_config, x)
            for x in ["api_base", "api_version", "api_type", "engine", "deployment_id"]
            if getattr(api_config, x) is not None
        }

        payload = {{"candidate_count": "n"}.get(k, k): v for k, v in payload.items()}
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]

        if api_config.api_type in ("azure", "azure_ad", "azuread"):
            deployment_id = envs.get("deployment_id")
            if envs.get("engine"):
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
            elif deployment_id is None:
                raise MlflowException(
                    "Either engine or deployment_id must be set for Azure OpenAI API",
                )
            payload = payload
        else:
            payload = {"model": endpoint, **payload}

        payload_with_envs = {**payload, **envs}

        try:
            resp = process_api_requests(
                [payload_with_envs],
                openai.ChatCompletion,
                api_token=api_token,
                throw_original_error=True,
                max_workers=1,
            )[0]
        except openai.error.AuthenticationError as e:
            raise MlflowException(
                f"Authentication Error for OpenAI. Error response:\n {e}",
                error_code=UNAUTHENTICATED,
            )
        except openai.error.InvalidRequestError as e:
            raise MlflowException(
                f"Invalid Request to OpenAI. Error response:\n {e}", error_code=BAD_REQUEST
            )
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
