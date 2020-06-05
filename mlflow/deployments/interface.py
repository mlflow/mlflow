from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.utils import experimental


listType = list
plugin_store = DeploymentPlugins(auto_register=True)


@experimental
def get_deploy_client(target_uri):
    """
    It calls the `get_deploy_client` method from the plugin and return an instance of a subclass of
    :py:class:`mlflow.deployments.BaseDeploymentClient` that can be used to deploy models to the
    specified target

    :param: target_uri: URI of target to deploy to. Run ``mlflow deployments --help`` via the CLI
                        for more information on supported deployment targets
    """
    # TODO: Maybe we should keep the separator as ":/" instead of "://"?
    target = target_uri.split("://")[0]
    return plugin_store[target].get_deploy_client(target_uri)


@experimental
def run_local(target, model_uri, flavor=None, config=None):
    """
    Deploys the specified model locally, for testing. This function calls the `run_local` function from
    the plugin and offload the task.

    :param target: Which target to use. This information is used to call the appropriate plugin
    :param model_uri: URI of model to deploy
    :param flavor: (optional) Model flavor to deploy. If unspecified, a default flavor will be chosen.
    :param config: (optional) Dict containing updated target-specific configuration for the deployment
    :return: None
    """
    return plugin_store[target].run_local(model_uri, flavor, config)


@experimental
def target_help(target):
    """
    Return a string containing detailed documentation on the current deployment target, to be displayed
    when users invoke the ``mlflow deployments help -t <target-name>`` CLI. This method should be defined
    within the module specified by the plugin author.
    The string should contain:
    * An explanation of target-specific fields in the ``config`` passed to ``create_deployment``,
      ``update_deployment``
    * How to specify a ``target_uri`` (e.g. for AWS SageMaker, ``target_uri``s have a scheme of
      "sagemaker://<aws-cli-profile-name>", where aws-cli-profile-name is the name of an AWS
      CLI profile https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
    * Any other target-specific details.

    :param target: Which target to use. This information is used to call the appropriate plugin
    """
    return plugin_store[target].target_help()
