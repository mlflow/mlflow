import inspect
from logging import Logger

from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.deployments.utils import get_deployments_target, parse_target_uri
from mlflow.exceptions import MlflowException

plugin_store = DeploymentPlugins()
plugin_store.register("sagemaker", "mlflow.sagemaker")

_logger = Logger(__name__)


def get_deploy_client(target_uri=None):
    """Returns a subclass of :py:class:`mlflow.deployments.BaseDeploymentClient` exposing standard
    APIs for deploying models to the specified target. See available deployment APIs
    by calling ``help()`` on the returned object or viewing docs for
    :py:class:`mlflow.deployments.BaseDeploymentClient`. You can also run
    ``mlflow deployments help -t <target-uri>`` via the CLI for more details on target-specific
    configuration options.

    Args:
        target_uri: Optional URI of target to deploy to. If no target URI is provided, then
            MLflow will attempt to get the deployments target set via `get_deployments_target()` or
            `MLFLOW_DEPLOYMENTS_TARGET` environment variable.

    .. code-block:: python
        :caption: Example

        from mlflow.deployments import get_deploy_client
        import pandas as pd

        client = get_deploy_client("redisai")
        # Deploy the model stored at artifact path 'myModel' under run with ID 'someRunId'. The
        # model artifacts are fetched from the current tracking server and then used for deployment.
        client.create_deployment("spamDetector", "runs:/someRunId/myModel")
        # Load a CSV of emails and score it against our deployment
        emails_df = pd.read_csv("...")
        prediction_df = client.predict_deployment("spamDetector", emails_df)
        # List all deployments, get details of our particular deployment
        print(client.list_deployments())
        print(client.get_deployment("spamDetector"))
        # Update our deployment to serve a different model
        client.update_deployment("spamDetector", "runs:/anotherRunId/myModel")
        # Delete our deployment
        client.delete_deployment("spamDetector")
    """
    if not target_uri:
        try:
            target_uri = get_deployments_target()
        except MlflowException:
            _logger.info(
                "No deployments target has been set. Please either set the MLflow deployments "
                "target via `mlflow.deployments.set_deployments_target()` or set the environment "
                "variable MLFLOW_DEPLOYMENTS_TARGET to the running deployment server's uri"
            )
            return None
    target = parse_target_uri(target_uri)
    plugin = plugin_store[target]
    for _, obj in inspect.getmembers(plugin):
        if inspect.isclass(obj):
            if issubclass(obj, BaseDeploymentClient) and not obj == BaseDeploymentClient:
                return obj(target_uri)


def run_local(target, name, model_uri, flavor=None, config=None):
    """Deploys the specified model locally, for testing. Note that models deployed locally cannot
    be managed by other deployment APIs (e.g. ``update_deployment``, ``delete_deployment``, etc).

    Args:
        target: Target to deploy to.
        name: Name to use for deployment
        model_uri: URI of model to deploy
        flavor: (optional) Model flavor to deploy. If unspecified, a default flavor
            will be chosen.
        config: (optional) Dict containing updated target-specific configuration for
            the deployment

    Returns:
        None
    """
    return plugin_store[target].run_local(name, model_uri, flavor, config)


def _target_help(target):
    """
    Return a string containing detailed documentation on the current deployment target,
    to be displayed when users invoke the ``mlflow deployments help -t <target-name>`` CLI.
    This method should be defined within the module specified by the plugin author.
    The string should contain:
    * An explanation of target-specific fields in the ``config`` passed to ``create_deployment``,
      ``update_deployment``
    * How to specify a ``target_uri`` (e.g. for AWS SageMaker, ``target_uri``s have a scheme of
      "sagemaker:/<aws-cli-profile-name>", where aws-cli-profile-name is the name of an AWS
      CLI profile https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
    * Any other target-specific details.

    Args:
        target: Which target to use. This information is used to call the appropriate plugin.
    """
    return plugin_store[target].target_help()
