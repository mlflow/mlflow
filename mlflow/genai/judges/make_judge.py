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
        instructions: Natural language instructions for evaluation. Must contain at least one
                      template variable: {{ inputs }}, {{ outputs }}, {{ expectations }},
                      or {{ trace }} to reference evaluation data. Custom variables are not
                      supported.
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


            # Create an agent that responds to questions
            def my_agent(question):
                # Simple toy agent that just echoes back
                return f"You asked about: {question}"


            # Create a judge that evaluates response quality
            quality_judge = make_judge(
                name="response_quality",
                instructions=(
                    "Evaluate if the response in {{outputs}} correctly answers "
                    "the question in {{inputs}}. The response should be accurate, "
                    "complete, and professional."
                ),
                model="openai:/gpt-4",
                aggregations=["mean", "max"],
            )

            # Get agent response
            question = "What is machine learning?"
            response = my_agent(question)

            # Evaluate the response
            result = quality_judge(
                inputs={"question": question},
                outputs={"response": response},
            )
            print(f"Score: {result.value}")
            print(f"Rationale: {result.rationale}")

        **Template Variable Behavior**:

        - ``{{inputs}}``, ``{{outputs}}``, ``{{expectations}}``: These variables are directly
          interpolated into the prompt. The dictionaries you pass are formatted and inserted
          as strings in the instruction template.

        - ``{{trace}}``: This variable has special behavior. Instead of interpolating the trace
          as JSON, the trace metadata is passed to an evaluation agent that fetches and analyzes
          the full trace details. This is useful for evaluating complex multi-step interactions
          that are logged as traces.

        .. code-block:: python

            # Example with trace evaluation
            trace_judge = make_judge(
                name="trace_analyzer",
                instructions=(
                    "Analyze the {{trace}} and determine if the agent's responses "
                    "were helpful and accurate throughout the conversation."
                ),
                model="openai:/gpt-4",
            )

            # Evaluate with expectations (must be dictionaries)
            result = correctness_judge(
                inputs={"question": "What is the capital of France?"},
                outputs={"answer": "The capital of France is Paris."},
                expectations={"expected_answer": "Paris"},
            )

            # Create a judge that evaluates based on trace context
            trace_judge = make_judge(
                name="trace_quality",
                instructions="Evaluate the overall quality of the {{ trace }} execution.",
                model="openai:/gpt-4",
            )

            # Use with search_traces() - evaluate each trace
            traces = mlflow.search_traces(experiment_ids=["1"], return_type="list")
            for trace in traces:
                feedback = trace_judge(trace=trace)
                print(f"Trace {trace.info.trace_id}: {feedback.value} - {feedback.rationale}")
    """

    if aggregations is None:
        aggregations = []
    validate_aggregations(aggregations)

    return InstructionsJudge(
        name=name, instructions=instructions, model=model, aggregations=aggregations
    )
