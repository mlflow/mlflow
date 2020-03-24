import click
from mlflow.utils import cli_args
from mlflow.deployments import interface
from mlflow.deployments.utils import parse_custom_arguments


deployment_target = click.option("--target", "-t", required=True, help="Deployment target")
deployment_id = click.option("--id", "_deployment_id", required=True,
                             help="Deployment ID for the deployment that needs to be deleted")
context_settings = dict(allow_extra_args=True, ignore_unknown_options=True,)


@click.group("deployments")
def commands():
    """
    Deploy MLflow models.  Downstream functions calls the plugin registered for the
    given target and pass the given arguments to it. Each of these functions also
    allows users to pass plugin specific arguments which will be processed by
    ``parse_custom_arguments`` function.
    
    MLflow will provide builtin support for some deployment targets eventually.
    However, Support for ANY targets is only available via MLflow plugins right now - see
    `community-plugins <https://mlflow.org/docs/latest/plugins.html#community-plugins>`_
    for a list of supported plugins. MLFlow also enables you to write custom plugins for
    deployment. For instructions on how to write and distribute your own plugin, checkout
    `here <https://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins>`_.
    """
    pass


@commands.command("create", context_settings=context_settings)
@parse_custom_arguments
@click.option("--flavor", "-f", help="Which flavor to be deployed. This will be auto "
                                     "inferred if it's not given")
@deployment_target
@cli_args.MODEL_URI
def create_cli(model_uri, target, flavor, **kwargs):
    """
    Create the deployment on the given target with the model from ``model_uri``.
    """
    deployment = interface.create(target, model_uri, flavor, **kwargs)
    # TODO: Support async here and everywhere requires
    click.echo("\n{} deployment {} is created".format(deployment.flavor, deployment.id))


@commands.command("delete", context_settings=context_settings)
@parse_custom_arguments
@deployment_id
@deployment_target
def delete_cli(target, _deployment_id, **kwargs):
    """
    Delete the deployment on the given target associated with the deployment id.
    """
    interface.delete(target, _deployment_id, **kwargs)
    click.echo("Deployment {} is deleted".format(_deployment_id))


@commands.command("update", context_settings=context_settings)
@parse_custom_arguments
@click.option("--rollback", help="Should the deployment be rolled back. Rollback is an "
                                 "option for some of the deployment targets but not for "
                                 "all. Make sure your deployment target supports rolling"
                                 " back", is_flag=True,)
@click.option("--model-uri", "-m", default=None, metavar="URI",
              help="URI to the model. A local path, a 'runs:/' URI, or a"
              " remote storage URI (e.g., an 's3://' URI). For more information"
              " about supported remote URIs for model artifacts, see"
              " https://mlflow.org/docs/latest/tracking.html"
              "#artifact-stores")  # optional model_uri
@deployment_id
@deployment_target
def update_cli(target, _deployment_id, rollback, model_uri, **kwargs):
    """
    Update the deployment associated with the deployment id at the given target with the new
    model. If rollback is True, it triggers rollback to previous version instead of updating
    with a new model
    """
    interface.update(target, _deployment_id, rollback=rollback, model_uri=model_uri, **kwargs)
    click.echo("Deployment {} is updated".format(_deployment_id))


@commands.command("list", context_settings=context_settings)
@parse_custom_arguments
@deployment_target
def list_cli(target, **kwargs):
    """
    List the all the deployment IDs from the target. These IDs can be used in delete, update,
    and describe APIs
    """
    ids = interface.list(target, **kwargs)
    click.echo("List of all deployments:\n{}".format(ids))


@commands.command("describe", context_settings=context_settings)
@parse_custom_arguments
@deployment_id
@deployment_target
def describe_cli(target, _deployment_id, **kwargs):
    """
    Fetch more details about the deployment associated with the given ID
    """
    desc = interface.describe(target, _deployment_id, **kwargs)
    click.echo(desc)
