import json
from typing import Literal

import click

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import get_all_scorers
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
    required=False,
    help="Experiment ID for which to list scorers. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)
@click.option(
    "--builtin",
    "-b",
    is_flag=True,
    default=False,
    help="List built-in scorers instead of registered scorers for an experiment.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' for formatted table (default) or 'json' for JSON format",
)
def list_scorers(
    experiment_id: str | None, builtin: bool, output: Literal["table", "json"]
) -> None:
    """
    List registered scorers for an experiment, or list all built-in scorers.

    \b
    Examples:

    .. code-block:: bash

        # List built-in scorers (table format)
        mlflow scorers list --builtin
        mlflow scorers list -b

        # List built-in scorers (JSON format)
        mlflow scorers list --builtin --output json

        # List registered scorers in table format (default)
        mlflow scorers list --experiment-id 123

        # List registered scorers in JSON format
        mlflow scorers list --experiment-id 123 --output json

        # Using environment variable for experiment ID
        export MLFLOW_EXPERIMENT_ID=123
        mlflow scorers list
    """
    # Validate mutual exclusivity
    if builtin and experiment_id:
        raise click.UsageError(
            "Cannot specify both --builtin and --experiment-id. "
            "Use --builtin to list built-in scorers or --experiment-id to list "
            "registered scorers for an experiment."
        )

    if not builtin and not experiment_id:
        raise click.UsageError(
            "Must specify either --builtin or --experiment-id. "
            "Use --builtin to list built-in scorers or --experiment-id to list "
            "registered scorers for an experiment."
        )

    # Get scorers based on mode
    scorers = get_all_scorers() if builtin else list_scorers_api(experiment_id=experiment_id)

    # Format scorer data for output
    scorer_data = [{"name": scorer.name, "description": scorer.description} for scorer in scorers]

    if output == "json":
        result = {"scorers": scorer_data}
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output format
        table = [[s["name"], s["description"] or ""] for s in scorer_data]
        click.echo(_create_table(table, headers=["Scorer Name", "Description"]))


@commands.command("register-llm-judge")
@click.option(
    "--name",
    "-n",
    type=click.STRING,
    required=True,
    help="Name for the judge scorer",
)
@click.option(
    "--instructions",
    "-i",
    type=click.STRING,
    required=True,
    help=(
        "Instructions for evaluation. Must contain at least one template variable: "
        "``{{ inputs }}``, ``{{ outputs }}``, ``{{ expectations }}``, or ``{{ trace }}``. "
        "See the make_judge documentation for variable interpretations."
    ),
)
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    required=False,
    help=(
        "Model identifier to use for evaluation (e.g., ``openai:/gpt-4``). "
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
@click.option(
    "--description",
    "-d",
    type=click.STRING,
    required=False,
    help="Description of what the judge evaluates.",
)
def register_llm_judge(
    name: str, instructions: str, model: str | None, experiment_id: str, description: str | None
) -> None:
    """
    Register an LLM judge scorer in the specified experiment.

    This command creates an LLM judge using natural language instructions and registers
    it in an experiment for use in evaluation workflows. The instructions must contain at
    least one template variable (``{{ inputs }}``, ``{{ outputs }}``, ``{{ expectations }}``,
    or ``{{ trace }}``) to define what the judge will evaluate.

    \b
    Examples:

    .. code-block:: bash

        # Register a basic quality judge
        mlflow scorers register-llm-judge -n quality_judge \\
            -i "Evaluate if {{ outputs }} answers {{ inputs }}. Return yes or no." -x 123

        # Register a judge with custom model
        mlflow scorers register-llm-judge -n custom_judge \\
            -i "Check whether {{ outputs }} is professional and formal. Rate pass, fail, or na" \\
            -m "openai:/gpt-4" -x 123

        # Register a judge with description
        mlflow scorers register-llm-judge -n quality_judge \\
            -i "Evaluate if {{ outputs }} answers {{ inputs }}. Return yes or no." \\
            -d "Evaluates response quality and relevance" -x 123

        # Using environment variable
        export MLFLOW_EXPERIMENT_ID=123
        mlflow scorers register-llm-judge -n my_judge \\
            -i "Check whether {{ outputs }} contains PII"
    """
    judge = make_judge(
        name=name,
        instructions=instructions,
        model=model,
        description=description,
        feedback_value_type=str,
    )
    registered_judge = judge.register(experiment_id=experiment_id)
    click.echo(
        f"Successfully created and registered judge scorer '{registered_judge.name}' "
        f"in experiment {experiment_id}"
    )
