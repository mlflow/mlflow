from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.telemetry.events import MakeJudgeEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@record_usage_event(MakeJudgeEvent)
def make_judge(
    name: str,
    instructions: str,
    model: str | None = None,
) -> Judge:
    """

    .. note::
        As of MLflow 3.4.0, this function is deprecated in favor of `mlflow.genai.make_judge`
        and may be removed in a future version.

    Create a custom MLflow judge instance.

    Args:
        name: The name of the judge
        instructions: Natural language instructions for evaluation. Must contain at least one
                      template variable: {{ inputs }}, {{ outputs }}, {{ expectations }},
                      or {{ trace }} to reference evaluation data. Custom variables are not
                      supported.
        model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")

    Returns:
        An InstructionsJudge instance configured with the provided parameters

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge

            # Create a judge that evaluates response quality using template variables
            quality_judge = make_judge(
                name="response_quality",
                instructions=(
                    "Evaluate if the response in {{ outputs }} correctly answers "
                    "the question in {{ inputs }}. The response should be accurate, "
                    "complete, and professional."
                ),
                model="openai:/gpt-4",
            )

            # Evaluate a response
            result = quality_judge(
                inputs={"question": "What is machine learning?"},
                outputs="ML is basically when computers learn stuff on their own",
            )

            # Create a judge that compares against expectations
            correctness_judge = make_judge(
                name="correctness",
                instructions=(
                    "Compare the {{ outputs }} against the {{ expectations }}. "
                    "Rate how well they match on a scale of 1-5."
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

            # Align a judge with human feedback
            aligned_judge = quality_judge.align(traces)

            # To see detailed optimization output during alignment, enable DEBUG logging:
            # import logging
            # logging.getLogger("mlflow.genai.judges.optimizers.simba").setLevel(logging.DEBUG)
    """

    return InstructionsJudge(name=name, instructions=instructions, model=model)
