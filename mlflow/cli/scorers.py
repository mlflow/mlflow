import json

import click

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.genai.scorers import list_scorers as list_scorers_api
from mlflow.utils.string_utils import _create_table


@click.group("scorers")
def commands():
    """
    Manage scorers, including LLM judges. To manage scorers associated with a tracking
    server, set the MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """


@commands.command("list")
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID for which to list scorers. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' for formatted table (default) or 'json' for JSON format",
)
def list_scorers(experiment_id: str, output: str) -> None:
    """
    List all registered scorers, including LLM judges, for the specified experiment.

    Examples:

    \b
    # List scorers in table format (default)
    mlflow scorers list --experiment-id 123

    \b
    # List scorers in JSON format
    mlflow scorers list --experiment-id 123 --output json

    \b
    # Using environment variable
    export MLFLOW_EXPERIMENT_ID=123
    mlflow scorers list
    """
    scorers = list_scorers_api(experiment_id=experiment_id)
    scorer_names = [scorer.name for scorer in scorers]

    if output == "json":
        result = {"scorers": scorer_names}
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output format
        table = [[name] for name in scorer_names]
        click.echo(_create_table(table, headers=["Scorer Name"]))
