import click

from mlflow.pipelines.utils import get_default_profile, _PIPELINE_PROFILE_ENV_VAR
from mlflow.pipelines import Pipeline
from mlflow.utils.annotations import experimental


@click.group("pipelines")
@experimental("command")
def commands():
    """
    Pipelines commands
    """
    pass


@commands.command()
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the pipeline step to run.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
@experimental("command")
def run(step, profile):
    """
    Run an individual step in the pipeline. If no step is specified, run all steps sequentially.
    """
    Pipeline(profile=profile).run(step)


@commands.command()
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The pipeline step whose execution cached output to be cleaned.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
@experimental("command")
def clean(step, profile):
    """
    Clean the cache associated with an individual step run. If the step is not specified, clean the
    entire pipeline cache.
    """
    Pipeline(profile=profile).clean(step)


@commands.command()
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The pipeline step to be inspected.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
@experimental("command")
def inspect(step, profile):
    """
    Inspect a step output. If no step is provided, visualize the full pipeline graph.
    """
    Pipeline(profile=profile).inspect(step)
