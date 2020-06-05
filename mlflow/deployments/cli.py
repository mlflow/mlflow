import click
from mlflow.utils import cli_args
from mlflow.deployments import interface
import sys
from mlflow.utils.logging_utils import eprint


def _user_args_to_dict(user_list, flag_name='P'):
    # TODO: This function is copied from mlflow.cli which is raising an issue when this
    #   is developed
    user_dict = {}
    for s in user_list:
        index = s.find("=")
        if index == -1:
            eprint("Invalid format for -%s parameter: '%s'. "
                   "Use -%s name=value." % (flag_name, s, flag_name))
            sys.exit(1)
        name = s[:index]
        value = s[index + 1:]
        if name in user_dict:
            eprint("Repeated parameter: '%s'" % name)
            sys.exit(1)
        user_dict[name] = value
    return user_dict


target_uri = click.option("--target-uri", "-t", "target", required=True,
                          help="Deployment target URI. Check the documentation/help for each "
                               "plugin to understand the target URI format the plugins expect. "
                               "For fetching the help, you can call "
                               "`mlflow deployments help <target-name>`")
deployment_name = click.option("--name", "name", required=True,
                               help="Name of the deployment")
parse_custom_arguments = click.option("--config", "-C", metavar="NAME=VALUE", multiple=True,
                                      help="Extra target-specific config for the model deployment,"
                                           " of the form -C name=value. See documentation/help for "
                                           "your deployment target for a list of supported config"
                                           " options.")
context_settings = dict(allow_extra_args=True, ignore_unknown_options=True,)


@click.group("deployments")
def commands():
    """
    [experimental] Deploy MLflow models to custom targets.

    To deploy to a custom target, you must first install an
    appropriate third-party Python plugin. See the list of known community-maintained plugins
    at https://mlflow.org/docs/latest/plugins.html#community-plugins.

    MLflow also enables you to write plugins for deployment to custom targets. For instructions on
    writing and distributing your own plugin, see
    https://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins.
    """
    pass


@commands.command("create", context_settings=context_settings)
@parse_custom_arguments
@deployment_name
@target_uri
@cli_args.MODEL_URI
@click.option("--flavor", "-f", help="Which flavor to be deployed. This will be auto "
                                     "inferred if it's not given")
def create_deployment(flavor, model_uri, target, name, config):
    """
    Deploy the model at ``model_uri`` to the specified target.

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`
    """
    config_dict = _user_args_to_dict(config, flag_name="C")
    client = interface.get_deploy_client(target)
    deployment = client.create_deployment(name, model_uri, flavor, config=config_dict)
    click.echo("\n{} deployment {} is created".format(deployment['flavor'],
                                                      deployment['name']))


@commands.command("update", context_settings=context_settings)
@parse_custom_arguments
@deployment_name
@target_uri
@click.option("--model-uri", "-m", default=None, metavar="URI",
              help="URI to the model. A local path, a 'runs:/' URI, or a"
                   " remote storage URI (e.g., an 's3://' URI). For more information"
                   " about supported remote URIs for model artifacts, see"
                   " https://mlflow.org/docs/latest/tracking.html"
                   "#artifact-stores")  # optional model_uri
@click.option("--flavor", "-f", help="Which flavor to be deployed. This will be auto "
                                     "inferred if it's not given")
def update_deployment(flavor, model_uri, target, name, config):
    """
    Update the deployment with ID `deployment_id` in the specified target.
    You can update the URI of the model and/or the flavor of the deployed model (in which case the
    model URI must also be specified).

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`.
    """
    config_dict = _user_args_to_dict(config, flag_name="C")
    client = interface.get_deploy_client(target)
    ret = client.update_deployment(name, model_uri=model_uri, flavor=flavor, config=config_dict)
    click.echo("Deployment {} is updated (with flavor {})".format(name, ret['flavor']))


@commands.command("delete", context_settings=context_settings)
@deployment_name
@target_uri
def delete_deployment(target, name):
    """
    Delete the deployment with ID `deployment_id` from the specified target.
    """
    client = interface.get_deploy_client(target)
    client.delete_deployment(name)
    click.echo("Deployment {} is deleted".format(name))


@commands.command("list", context_settings=context_settings)
@target_uri
def list_deployment(target):
    """
    List the IDs of all model deployments in the specified target. These IDs can be used with
    the `delete`, `update`, and `get` commands.
    """
    client = interface.get_deploy_client(target)
    ids = client.list_deployments()
    click.echo("List of all deployments:\n{}".format(ids))


@commands.command("get", context_settings=context_settings)
@deployment_name
@target_uri
def get_deployment(target, name):
    """
    Print a detailed description of the deployment with ID ``deployment_id`` in the specified
    target.
    """
    client = interface.get_deploy_client(target)
    desc = client.get_deployment(name)
    for key, val in desc.items():
        click.echo("{}: {}".format(key, val))
    click.echo('\n')


@commands.command("get", context_settings=context_settings)
@click.argument("target")
def target_help(target):
    """
    Specific help command for deployment plugins. This will call the `target_help` function from
    the plugin to display the help string specific for each plugin
    """
    click.echo(interface.target_help(target))
