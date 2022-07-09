import click

from mlflow.pipelines.utils import _PIPELINE_PROFILE_ENV_VAR
from mlflow.pipelines import Pipeline
from mlflow.utils.annotations import experimental

_CLI_ARG_PIPELINE_PROFILE = click.option(
    "--profile",
    "-p",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=None,
    required=True,
    help=(
        "The name of the pipeline profile to use. Profiles customize the configuration of"
        " one or more pipeline steps, and pipeline executions with different profiles often"
        " produce different results."
    ),
)


_CLI_ARG_PIPELINE_FILE_NAME = click.option(
    "--pipeline_file_name",
    type=click.STRING,
    default="pipeline.yaml",
    required=False,
    help=(
        "The name of the pipeline YAML file to use. If not provided, will default to" +
        " 'pipeline.yaml'"
    ),    
)


@click.group("pipelines")
def commands():
    """
    Run MLflow Pipelines and inspect pipeline results.
    """
    pass


@commands.command(short_help="Run the full pipeline or a particular pipeline step.")
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the pipeline step to run.",
)
@_CLI_ARG_PIPELINE_PROFILE
@_CLI_ARG_PIPELINE_FILE_NAME
@experimental("command")
def run(step, profile, pipeline_file_name):
    """
    Run the full pipeline, or run a particular pipeline step if specified, producing
    outputs and displaying a summary of results upon completion.
    """
    Pipeline(profile=profile, pipeline_file_name=pipeline_file_name).run(step)


@commands.command(
    short_help=(
        "Remove all pipeline outputs from the cache, or remove the cached outputs of"
        " a particular pipeline step."
    )
)
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the pipeline step for which to remove cached outputs.",
)
@_CLI_ARG_PIPELINE_PROFILE
@_CLI_ARG_PIPELINE_FILE_NAME
@experimental("command")
def clean(step, profile, pipeline_file_name):
    """
    Remove all pipeline outputs from the cache, or remove the cached outputs of a particular
    pipeline step if specified. After cached outputs are cleaned for a particular step, the step
    will be re-executed in its entirety the next time it is run.
    """
    Pipeline(profile=profile, pipeline_file_name=pipeline_file_name).clean(step)


@commands.command(
    short_help=(
        "Display an overview of the pipeline graph or a summary of results from a particular step."
    )
)
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the pipeline step to inspect.",
)
@_CLI_ARG_PIPELINE_PROFILE
@_CLI_ARG_PIPELINE_FILE_NAME
@experimental("command")
def inspect(step, profile, pipeline_file_name):
    """
    Display a visual overview of the pipeline graph, or display a summary of results from a
    particular pipeline step if specified. If the specified step has not been executed,
    nothing is displayed.
    """
    Pipeline(profile=profile, pipeline_file_name=pipeline_file_name).inspect(step)
