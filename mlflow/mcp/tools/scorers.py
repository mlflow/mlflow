from typing import Annotated

from pydantic import Field

from mlflow.cli.scorers import _builtin_scorer_catalog
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import list_scorers as list_registered_scorers
from mlflow.mcp.tools._args import DeprecatedOutput, as_json_object
from mlflow.mcp.tools._types import RegisteredScorer, ScorerInfo, ScorerList


def list_scorers(
    experiment_id: Annotated[
        str | None,
        Field(description="Experiment to list registered scorers of. Give this or builtin."),
    ] = None,
    builtin: Annotated[
        bool,
        Field(
            description="List the built-in scorers instead, with the columns they require, "
            "whether they are session-level and the arguments needed to construct them."
        ),
    ] = False,
    output: DeprecatedOutput = None,
) -> ScorerList:
    """List the scorers registered in an experiment, or every built-in scorer."""
    if builtin == (experiment_id is not None):
        raise MlflowException.invalid_parameter_value(
            "Must specify exactly one of builtin or experiment_id."
        )
    if builtin:
        return ScorerList(scorers=[ScorerInfo(**entry) for entry in _builtin_scorer_catalog()])
    return ScorerList(
        scorers=[
            ScorerInfo(name=scorer.name, description=scorer.description)
            for scorer in list_registered_scorers(experiment_id=experiment_id)
        ]
    )


def register_llm_judge_scorer(
    name: Annotated[str, Field(description="Name for the judge scorer.")],
    instructions: Annotated[
        str,
        Field(
            description="Evaluation instructions. Must contain at least one template variable: "
            "{{ inputs }}, {{ outputs }}, {{ expectations }} or {{ trace }}."
        ),
    ],
    experiment_id: Annotated[str, Field(description="Experiment to register the judge in.")],
    model: Annotated[
        str | None,
        Field(
            description="Model used for evaluation, e.g. 'openai:/gpt-4'. The default judge "
            "model when omitted."
        ),
    ] = None,
    description: Annotated[
        str | None, Field(description="Description of what the judge evaluates.")
    ] = None,
    base_url: Annotated[
        str | None,
        Field(
            description="Base URL to route LLM requests through. Not persisted with the "
            "registered judge."
        ),
    ] = None,
    extra_headers: Annotated[
        dict[str, str] | str | None,
        Field(
            description="Additional HTTP headers for the LLM provider, as an object or a JSON "
            "string. Not persisted with the registered judge."
        ),
    ] = None,
) -> RegisteredScorer:
    """Create an LLM judge from natural-language instructions and register it in an experiment."""
    headers = as_json_object(extra_headers, "extra_headers")
    if headers is not None and not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise MlflowException.invalid_parameter_value(
            "`extra_headers` keys and values must all be strings."
        )
    judge = make_judge(
        name=name,
        instructions=instructions,
        model=model,
        description=description,
        feedback_value_type=str,
        base_url=base_url,
        extra_headers=headers,
    )
    registered = judge.register(experiment_id=experiment_id)
    return RegisteredScorer(name=registered.name, experiment_id=experiment_id)
