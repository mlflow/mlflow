import click
from mlflow.utils import cli_args
from mlflow.deployments import interface
from mlflow.deployments.utils import parse_custom_arguments


deployment_target = click.option("--target", "-t", required=True, help="Deployment target")
deployment_id = click.option("--id", "_deployment_id", required=True,
                             help="ID of the deployment to delete")
context_settings = dict(allow_extra_args=True, ignore_unknown_options=True,)


@click.group("deployments")
def commands():
    """
    [experimental] Deploy MLflow models to custom targets.

    To deploy to a custom target, you must first install an
    appropriate third-party Python plugin. See the list of known community-maintained plugins
    at <https://mlflow.org/docs/latest/plugins.html#community-plugins>`_.

    MLflow also enables you to write plugins for deployment to custom targets. For instructions on
    writing and distributing your own plugin, see
    `<https://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins>`_.
    """
    pass


@commands.command("create", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
@cli_args.MODEL_URI
@click.option("--flavor", "-f", help="Which flavor to be deployed. This will be auto "
                                     "inferred if it's not given")
def create_cli(flavor, model_uri, target, **kwargs):
    """
    Deploy the model at ``model_uri`` to the specified target.

    Additional plugin-specific arguments may also be passed to this command, via syntax like
    `--param-name value`
    """
    deployment = interface.create_deployment(target, model_uri, flavor, **kwargs)
    # TODO: Support async here and everywhere requires
    click.echo("\n{} deployment {} is created".format(deployment['flavor'],
                                                      deployment['deployment_id']))


@commands.command("delete", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
@deployment_id
def delete_cli(_deployment_id, target, **kwargs):
    """
    Delete the deployment with ID `deployment_id` from the specified target.

    Additional plugin-specific arguments may also be passed to this command, via syntax like
    `--param-name value`.
    """
    interface.delete_deployment(target, _deployment_id, **kwargs)
    click.echo("Deployment {} is deleted".format(_deployment_id))


@commands.command("update", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
@deployment_id
@click.option("--model-uri", "-m", default=None, metavar="URI",
              help="URI to the model. A local path, a 'runs:/' URI, or a"
              " remote storage URI (e.g., an 's3://' URI). For more information"
              " about supported remote URIs for model artifacts, see"
              " https://mlflow.org/docs/latest/tracking.html"
              "#artifact-stores")  # optional model_uri
@click.option("--flavor", "-f", help="Which flavor to be deployed. This will be auto "
                                     "inferred if it's not given")
def update_cli(flavor, model_uri, _deployment_id, target, **kwargs):
    """
    Update the deployment with ID `deployment_id` in the specified target.

    Additional plugin-specific arguments may also be passed to this command, via syntax like
    `--param-name value`.
    """
    ret = interface.update_deployment(target, _deployment_id,
                                      model_uri=model_uri, flavor=flavor, **kwargs)
    click.echo("Deployment {} is updated (with flavor {})".format(_deployment_id,
                                                                  ret['flavor']))


@commands.command("list", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
def list_cli(target, **kwargs):
    """
    List the IDs of all model deployments in the specified target. These IDs can be used with
    the `delete`, `update`, and `describe` commands.

    Additional plugin-specific arguments may also be passed to this command, via syntax like
    `--param-name value`.
    """
    ids = interface.list_deployments(target, **kwargs)
    click.echo("List of all deployments:\n{}".format(ids))


@commands.command("describe", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
@deployment_id
def describe_cli(_deployment_id, target, **kwargs):
    """
    Print a detailed description of the deployment with ID ``deployment_id`` in the specified
    target.

    Additional plugin-specific arguments may also be passed to this command, via syntax like
    `--param-name value`.
    """
    desc = interface.describe_deployment(target, _deployment_id, **kwargs)
    for key, val in desc.items():
        click.echo("{}: {}".format(key, val))
    click.echo('\n')
