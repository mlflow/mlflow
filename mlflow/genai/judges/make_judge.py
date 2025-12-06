from typing import Any, Literal, get_args, get_origin

from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.telemetry.events import MakeJudgeEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils.annotations import experimental


def _validate_feedback_value_type(feedback_value_type: Any) -> None:
    """
    Validate that feedback_value_type is one of the supported types for serialization.

    Supported types match FeedbackValueType:
    - PbValueType: int, float, str, bool
    - Literal types with PbValueType values
    - dict[str, PbValueType]
    - list[PbValueType]
    """

    from mlflow.entities.assessment import PbValueType

    # Check for basic PbValueType (float, int, str, bool)
    pb_value_types = get_args(PbValueType)
    if feedback_value_type in pb_value_types:
        return

    # Check for Literal type
    origin = get_origin(feedback_value_type)
    if origin is Literal:
        # Validate that all literal values are of PbValueType
        literal_values = get_args(feedback_value_type)
        for value in literal_values:
            if not isinstance(value, pb_value_types):
                from mlflow.exceptions import MlflowException

                raise MlflowException.invalid_parameter_value(
                    "The `feedback_value_type` argument does not support a Literal type"
                    f"with non-primitive types, but got {type(value).__name__}. "
                    f"Literal values must be str, int, float, or bool."
                )
        return

    # Check for dict[str, PbValueType]
    if origin is dict:
        args = get_args(feedback_value_type)
        if len(args) == 2:
            key_type, value_type = args
            # Key must be str
            if key_type != str:
                from mlflow.exceptions import MlflowException

                raise MlflowException.invalid_parameter_value(
                    f"dict key type must be str, got {key_type}"
                )
            # Value must be a PbValueType
            if value_type not in pb_value_types:
                from mlflow.exceptions import MlflowException

                raise MlflowException.invalid_parameter_value(
                    "The `feedback_value_type` argument does not support a dict type"
                    f"with non-primitive values, but got {value_type.__name__}"
                )
            return

    # Check for list[PbValueType]
    if origin is list:
        args = get_args(feedback_value_type)
        if len(args) == 1:
            element_type = args[0]
            # Element must be a PbValueType
            if element_type not in pb_value_types:
                from mlflow.exceptions import MlflowException

                raise MlflowException.invalid_parameter_value(
                    "The `feedback_value_type` argument does not support a list type"
                    f"with non-primitive values, but got {element_type.__name__}"
                )
            return

    # If we get here, it's an unsupported type
    from mlflow.exceptions import MlflowException

    raise MlflowException.invalid_parameter_value(
        f"Unsupported feedback_value_type: {feedback_value_type}. "
        f"Supported types (FeedbackValueType): str, int, float, bool, Literal[...], "
        f"as well as a dict and list of these types. "
        f"Pydantic BaseModel types are not supported."
    )


@experimental(version="3.4.0")
@record_usage_event(MakeJudgeEvent)
def make_judge(
    name: str,
    instructions: str,
    model: str | None = None,
    description: str | None = None,
    feedback_value_type: Any = None,
    inference_params: dict[str, Any] | None = None,
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
                      {{ conversation }}, or {{ trace }} to reference evaluation data. Custom
                      variables are not supported.
                      Note: {{ conversation }} can only coexist with {{ expectations }}.
                      It cannot be used together with {{ inputs }}, {{ outputs }}, or {{ trace }}.
        model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")
        description: A description of what the judge evaluates
        feedback_value_type: Type specification for the 'value' field in the Feedback
                        object. The judge will use structured outputs to enforce this type.
                        If unspecified, the feedback value type is determined by the judge.
                        It is recommended to explicitly specify the type.

                        Supported types (matching FeedbackValueType):

                        - int: Integer ratings (e.g., 1-5 scale)
                        - float: Floating point scores (e.g., 0.0-1.0)
                        - str: Text responses
                        - bool: Yes/no evaluations
                        - Literal[values]: Enum-like choices (e.g., Literal["good", "bad"])
                        - dict[str, int | float | str | bool]: Dictionary with string keys and
                          int, float, str, or bool values.
                        - list[int | float | str | bool]: List of int, float, str, or bool values

                        Note: Pydantic BaseModel types are not supported.
        inference_params: Optional dictionary of inference parameters to pass to the model
                        (e.g., temperature, top_p, max_tokens). These parameters allow
                        fine-grained control over the model's behavior during evaluation.
                        For example, setting a lower temperature can produce more
                        deterministic and reproducible evaluation results.

    Returns:
        An InstructionsJudge instance configured with the provided parameters

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge
            from typing import Literal

            # Create a judge that evaluates response quality using template variables
            quality_judge = make_judge(
                name="response_quality",
                instructions=(
                    "Evaluate if the response in {{ outputs }} correctly answers "
                    "the question in {{ inputs }}. The response should be accurate, "
                    "complete, and professional."
                ),
                model="openai:/gpt-4",
                feedback_value_type=Literal["yes", "no"],
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
                feedback_value_type=int,
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
                feedback_value_type=Literal["good", "needs_improvement"],
            )

            # Use with search_traces() - evaluate each trace
            traces = mlflow.search_traces(experiment_ids=["1"], return_type="list")
            for trace in traces:
                feedback = trace_judge(trace=trace)
                print(f"Trace {trace.info.trace_id}: {feedback.value} - {feedback.rationale}")

            # Create a multi-turn judge that detects user frustration
            frustration_judge = make_judge(
                name="user_frustration",
                instructions=(
                    "Analyze the {{ conversation }} to detect signs of user frustration. "
                    "Look for indicators such as repeated questions, negative language, "
                    "or expressions of dissatisfaction."
                ),
                model="openai:/gpt-4",
                feedback_value_type=Literal["frustrated", "not frustrated"],
            )

            # Evaluate a multi-turn conversation using session traces
            session = mlflow.search_traces(
                experiment_ids=["1"],
                filter_string="metadata.`mlflow.trace.session` = 'session_123'",
                return_type="list",
            )
            result = frustration_judge(session=session)

            # Align a judge with human feedback
            aligned_judge = quality_judge.align(traces)

            # To see detailed optimization output during alignment, enable DEBUG logging:
            # import logging
            # logging.getLogger("mlflow.genai.judges.optimizers.simba").setLevel(logging.DEBUG)
    """
    # Default feedback_value_type to str if not specified (consistent with MLflow <= 3.5.x)
    # TODO: Implement logic to allow the LLM to choose the appropriate value type if not specified
    if feedback_value_type is None:
        feedback_value_type = str

    _validate_feedback_value_type(feedback_value_type)

    return InstructionsJudge(
        name=name,
        instructions=instructions,
        model=model,
        description=description,
        feedback_value_type=feedback_value_type,
        inference_params=inference_params,
    )
