import inspect
from six.moves import urllib
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.exceptions import MlflowException


plugin_store = DeploymentPlugins()


def get_uri_scheme(uri):
    # TODO: replace it with `mflow.utils.uri.get_uri_scheme` once verified
    uri = urllib.parse.urlparse(uri)
    if not uri.scheme:
        if uri.path:
            # uri = 'target_name' (without :/<path>)
            return uri.path
        raise MlflowException(
            "Not a proper deployment URI: %s. " % uri +
            "Deployment URIs must be of the form 'target:/server/details'")
    return uri.scheme


def get_deploy_client(target_uri):
    """
    It fetches all the classes inside the plugin and check for a subclass of
    :py:class:`mlflow.deployments.BaseDeploymentClient` that can be used to deploy
    models to the specified target

    .. Note::
        The plugin should only have one child class of
        :py:class:`mlflow.deployments.BaseDeploymentClient` else it throws.

    :param: target_uri: URI of target to deploy to. Run ``mlflow deployments --help`` via the CLI
                        for more information on supported deployment targets
    """
    target = get_uri_scheme(target_uri)
    plugin = plugin_store[target]
    for _, obj in inspect.getmembers(plugin):
        if issubclass(obj, BaseDeploymentClient) and not obj == BaseDeploymentClient:
            return obj(target_uri)


def run_local(target, name, model_uri, flavor=None, config=None):
    """
    Deploys the specified model locally, for testing. This function calls the `run_local` function
    from the plugin and offload the task.

    :param target: Which target to use. This information is used to call the appropriate plugin
    :param name:  Unique name to use for deployment. If another deployment exists with the same
                  name, this function will raise a
                  `:py:class:mlflow.exceptions.MlflowException`
    :param model_uri: URI of model to deploy
    :param flavor: (optional) Model flavor to deploy. If unspecified, a default flavor
                   will be chosen.
    :param config: (optional) Dict containing updated target-specific configuration for
                   the deployment
    :return: None
    """
    return plugin_store[target].run_local(name, model_uri, flavor, config)


def target_help(target):
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

    :param target: Which target to use. This information is used to call the appropriate plugin
    """
    return plugin_store[target].target_help()
