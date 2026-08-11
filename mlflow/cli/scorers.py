import json
from typing import Any, Literal

import click
from pydantic_core import PydanticUndefined

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import list_scorers as list_scorers_api
from mlflow.genai.scorers.builtin_scorers import (
    BuiltInScorer,
    SessionLevelScorer,
    _get_all_concrete_builtin_scorers,
)
from mlflow.mcp.decorator import mlflow_mcp
from mlflow.utils.string_utils import _create_table

# Fields shared by all built-in scorers (name, description, required_columns,
# inference_params, extra_headers). Excluded when deriving the constructor arguments a
# specific scorer requires, so only scorer-specific required fields (e.g. Guidelines'
# ``guidelines``) are reported.
_SHARED_SCORER_FIELDS = set(BuiltInScorer.model_fields)


def _builtin_scorer_catalog() -> list[dict[str, Any]]:
    """Describe every built-in scorer by reading its class, without instantiating it.

    ``get_all_scorers()`` only returns scorers instantiable with no arguments, so
    scorers with required constructor arguments (``Guidelines``, ``RegexMatch``,
    ``ResponseLength``, ``ConversationalGuidelines``) never appear. Reading fields off
    the class instead surfaces all of them, along with the data contract and the
    arguments needed to construct each one.

    Reports the statically declared requirements only. Some scorers enforce
    additional, alternative, or conditional requirements at construction or evaluation
    time that this flat view does not represent:

    - ``required_columns`` reflects the field default; scorers such as ``correctness``,
      ``equivalence``, and ``expectations_guidelines`` require one of
      ``expectations/expected_response`` or ``expectations/expected_facts`` (enforced in
      ``validate_columns()``).
    - ``required_args`` lists individually required constructor fields; it does not
      capture one-of requirements such as ``response_length`` needing at least one of
      ``min_length`` or ``max_length`` (enforced by a model validator).
    """
    catalog = []
    for cls in _get_all_concrete_builtin_scorers():
        fields = cls.model_fields
        required_columns_field = fields.get("required_columns")
        required_columns = (
            required_columns_field.default
            if required_columns_field is not None
            and required_columns_field.default is not PydanticUndefined
            else None
        )
        catalog.append({
            "name": fields["name"].default,
            "description": fields["description"].default,
            "required_columns": sorted(required_columns) if required_columns else [],
            "session_level": issubclass(cls, SessionLevelScorer),
            "required_args": sorted(
                name
                for name, field in fields.items()
                if field.is_required() and name not in _SHARED_SCORER_FIELDS
            ),
        })
    return sorted(catalog, key=lambda s: s["name"])


class DictParamType(click.ParamType):
    name = "dict"

    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            example = '{"key": "value"}'
            self.fail(
                f"Invalid JSON. Expected a JSON object, e.g. '{example}'.",
                param,
                ctx,
            )
        if not isinstance(parsed, dict):
            self.fail("Expected a JSON object (dict), not an array or scalar.", param, ctx)
        for k, v in parsed.items():
            if not isinstance(k, str) or not isinstance(v, str):
                self.fail(
                    f"Keys and values must all be strings, "
                    f"got key={k!r} ({type(k).__name__}), value={v!r} ({type(v).__name__}).",
                    param,
                    ctx,
                )
        return parsed


@click.group("scorers")
def commands():
    """
    Manage scorers, including LLM judges. To manage scorers associated with a tracking
    server, set the MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """


@commands.command("list")
@mlflow_mcp(tool_name="list_scorers")
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

    With ``--builtin``, the output describes every built-in scorer: the columns it
    requires, whether it is session-level, and any arguments needed to construct it
    (e.g. ``Guidelines`` needs ``guidelines``). Scorers with required arguments are
    included, unlike ``get_all_scorers()``, which only returns those instantiable with
    defaults.

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

    if builtin:
        scorer_data = _builtin_scorer_catalog()
        if output == "json":
            click.echo(json.dumps({"scorers": scorer_data}, indent=2))
        else:
            table = [
                [
                    s["name"],
                    ", ".join(s["required_columns"]) or "-",
                    "yes" if s["session_level"] else "no",
                    ", ".join(s["required_args"]) or "-",
                    s["description"] or "",
                ]
                for s in scorer_data
            ]
            click.echo(
                _create_table(
                    table,
                    headers=["Scorer Name", "Requires", "Session", "Args", "Description"],
                )
            )
        return

    scorers = list_scorers_api(experiment_id=experiment_id)
    scorer_data = [{"name": scorer.name, "description": scorer.description} for scorer in scorers]
    if output == "json":
        click.echo(json.dumps({"scorers": scorer_data}, indent=2))
    else:
        table = [[s["name"], s["description"] or ""] for s in scorer_data]
        click.echo(_create_table(table, headers=["Scorer Name", "Description"]))


@commands.command("register-llm-judge")
@mlflow_mcp(tool_name="register_llm_judge_scorer")
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
@click.option(
    "--base-url",
    type=click.STRING,
    required=False,
    help=(
        "Base URL to route requests through. Useful for enterprise environments "
        "requiring LLM access through internal gateways or security proxies. "
        "Note: This value is not persisted when the judge is registered."
    ),
)
@click.option(
    "--extra-headers",
    type=DictParamType(),
    required=False,
    help=(
        "JSON string of additional HTTP headers to include in requests to the LLM provider. "
        'Example: \'{{"X-API-Key": "secret"}}\'. '
        "Note: This value is not persisted when the judge is registered."
    ),
)
def register_llm_judge(
    name: str,
    instructions: str,
    model: str | None,
    experiment_id: str,
    description: str | None,
    base_url: str | None,
    extra_headers: dict[str, str] | None,
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
        base_url=base_url,
        extra_headers=extra_headers,
    )
    registered_judge = judge.register(experiment_id=experiment_id)
    click.echo(
        f"Successfully created and registered judge scorer '{registered_judge.name}' "
        f"in experiment {experiment_id}"
    )
