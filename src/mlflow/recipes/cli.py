import click

from mlflow.environment_variables import MLFLOW_RECIPES_PROFILE
from mlflow.recipes import Recipe

_CLI_ARG_RECIPE_PROFILE = click.option(
    "--profile",
    "-p",
    envvar=MLFLOW_RECIPES_PROFILE.name,
    type=click.STRING,
    default=None,
    required=True,
    help=(
        "The name of the recipe profile to use. Profiles customize the configuration of"
        " one or more recipe steps, and recipe executions with different profiles often"
        " produce different results."
    ),
)


@click.group("recipes")
def commands():
    """
    Run MLflow Recipes and inspect recipe results.
    """


@commands.command(short_help="Run the full recipe or a particular recipe step.")
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the recipe step to run.",
)
@_CLI_ARG_RECIPE_PROFILE
def run(step, profile):
    """
    Run the full recipe, or run a particular recipe step if specified, producing
    outputs and displaying a summary of results upon completion.
    """
    Recipe(profile=profile).run(step)


@commands.command(
    short_help=(
        "Remove all recipe outputs from the cache, or remove the cached outputs of"
        " a particular recipe step."
    )
)
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the recipe step for which to remove cached outputs.",
)
@_CLI_ARG_RECIPE_PROFILE
def clean(step, profile):
    """
    Remove all recipe outputs from the cache, or remove the cached outputs of a particular
    recipe step if specified. After cached outputs are cleaned for a particular step, the step
    will be re-executed in its entirety the next time it is run.
    """
    Recipe(profile=profile).clean(step)


@commands.command(
    short_help=(
        "Display an overview of the recipe graph or a summary of results from a particular step."
    )
)
@click.option(
    "--step",
    "-s",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the recipe step to inspect.",
)
@_CLI_ARG_RECIPE_PROFILE
def inspect(step, profile):
    """
    Display a visual overview of the recipe graph, or display a summary of results from a
    particular recipe step if specified. If the specified step has not been executed,
    nothing is displayed.
    """
    Recipe(profile=profile).inspect(step)


@commands.command(short_help=("Get the location of an artifact output from the recipe."))
@click.option(
    "--artifact",
    "-a",
    type=click.STRING,
    default=None,
    required=True,
    help="The name of the artifact to retrieve.",
)
@_CLI_ARG_RECIPE_PROFILE
def get_artifact(profile, artifact):
    """
    Get the location of an artifact output from the recipe.
    """
    artifact_location = Recipe(profile=profile)._get_artifact(artifact).path()
    click.echo(artifact_location)
