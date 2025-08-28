from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers.base import AggregationFunc
from mlflow.genai.scorers.validation import validate_aggregations
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def make_judge(
    name: str,
    instructions: str,
    model: str | None = None,
    aggregations: list[str | AggregationFunc] | None = None,
) -> Judge:
    """
    Create a custom MLflow judge instance.

    Args:
        name: The name of the judge
        instructions: Natural language instructions for evaluation
        model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")
        aggregations: List of aggregation functions to apply. Can be strings from
                      ["min", "max", "mean", "median", "variance", "p90"] or callable functions.
                      Defaults to [] (no aggregations) if not specified.

    Returns:
        An InstructionsJudge instance configured with the provided parameters

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge

            # Create a judge that evaluates response formality
            formality_judge = make_judge(
                name="formality_checker",
                instructions="The response should be formal and professional",
                model="openai:/gpt-4",
                aggregations=["mean", "max"],
            )

            # Evaluate a response
            result = formality_judge(
                inputs={"question": "What is machine learning?"},
                outputs="ML is basically when computers learn stuff on their own",
            )
    """

    if aggregations is None:
        aggregations = []

    validate_aggregations(aggregations)

    return InstructionsJudge(
        name=name, instructions=instructions, model=model, aggregations=aggregations
    )
