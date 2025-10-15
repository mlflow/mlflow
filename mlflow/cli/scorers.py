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


@commands.command("create-judge")
@click.option(
    "--name",
    "-n",
    type=click.STRING,
    required=True,
    help="Name for the judge scorer",
)
@click.option(
    "--prompt",
    "-p",
    type=click.STRING,
    required=True,
    help=(
        "Natural language instructions for evaluation. Must contain at least one "
        "template variable: {{ inputs }}, {{ outputs }}, {{ expectations }}, or {{ trace }}."
    ),
)
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    required=False,
    help=(
        "Model identifier to use for evaluation (e.g., 'openai:/gpt-4'). "
        "If not provided, uses the default model."
    ),
)
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID to register the judge in. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)
def create_judge(name: str, prompt: str, model: str | None, experiment_id: str) -> None:
    """
    Create and register a judge scorer for the specified experiment.

    Examples:

    \b
    # Create a basic quality judge
    mlflow scorers create-judge -n quality_judge \\
        -p "Evaluate if {{ outputs }} answers {{ inputs }}. Return yes or no." -x 123

    \b
    # Create a judge with custom model
    mlflow scorers create-judge -n custom_judge \
        -p "Check whether {{ outputs }} is professional and formal. \
            Rate pass, fail, or na" -m "openai:/gpt-4" -x 123

    \b
    # Using environment variable
    export MLFLOW_EXPERIMENT_ID=123
    mlflow scorers create-judge -n my_judge -p "Check whether {{ outputs }} contains PII"
    """
    from mlflow.exceptions import MlflowException
    from mlflow.genai.judges import make_judge

    try:
        judge = make_judge(name=name, instructions=prompt, model=model)
        registered_judge = judge.register(experiment_id=experiment_id)
        click.echo(
            f"Successfully created and registered judge scorer '{registered_judge.name}' "
            f"in experiment {experiment_id}"
        )
    except MlflowException as e:
        raise click.ClickException(str(e))
