import logging
import math
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pydantic

from mlflow.genai.judges.prompts.equivalence import EQUIVALENCE_PROMPT_INSTRUCTIONS

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import judges
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.constants import _AFFIRMATIVE_VALUES, _NEGATIVE_VALUES
from mlflow.genai.judges.prompts.context_sufficiency import CONTEXT_SUFFICIENCY_PROMPT_INSTRUCTIONS
from mlflow.genai.judges.prompts.correctness import CORRECTNESS_PROMPT_INSTRUCTIONS
from mlflow.genai.judges.prompts.groundedness import GROUNDEDNESS_PROMPT_INSTRUCTIONS
from mlflow.genai.judges.prompts.guidelines import GUIDELINES_PROMPT_INSTRUCTIONS
from mlflow.genai.judges.prompts.relevance_to_query import RELEVANCE_TO_QUERY_PROMPT_INSTRUCTIONS
from mlflow.genai.judges.utils import (
    CategoricalRating,
    get_chat_completions_with_structured_output,
    get_default_model,
    invoke_judge_model,
)
from mlflow.genai.scorers.base import (
    _SERIALIZATION_VERSION,
    ScorerKind,
    SerializedScorer,
)
from mlflow.genai.utils.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring
from mlflow.utils.uri import is_databricks_uri

GENAI_CONFIG_NAME = "databricks-agent"


@dataclass
class FieldExtractionConfig:
    messages: list["ChatMessage"]
    schema: type[pydantic.BaseModel]


@dataclass
class ExtractedFields:
    inputs: Any | None = None
    outputs: Any | None = None
    expectations: dict[str, Any] | None = None


def _construct_field_extraction_config(
    needs_inputs: bool,
    needs_outputs: bool,
) -> FieldExtractionConfig:
    """
    Construct field extraction configuration with messages and schema.

    Args:
        needs_inputs: Whether inputs field needs extraction.
        needs_outputs: Whether outputs field needs extraction.

    Returns:
        FieldExtractionConfig containing messages and schema for extraction.
    """
    from mlflow.types.llm import ChatMessage

    extraction_tasks = []
    schema_fields = {}

    if needs_inputs:
        extraction_tasks.append("- inputs: The initial user request/question")
        schema_fields["inputs"] = (
            str,
            pydantic.Field(description="The user's original request"),
        )

    if needs_outputs:
        extraction_tasks.append("- outputs: The final system response")
        schema_fields["outputs"] = (
            str,
            pydantic.Field(description="The system's final response"),
        )

    schema = pydantic.create_model("ExtractionSchema", **schema_fields)

    messages = [
        ChatMessage(
            role="system",
            content=(
                "Extract the following fields from the trace.\n"
                "Use the provided tools to examine the trace's spans to find:\n"
                + "\n".join(extraction_tasks)
                + "\n\nReturn the result as JSON."
            ),
        ),
        ChatMessage(
            role="user",
            content="Use the tools to find the required fields, then return them as JSON.",
        ),
    ]

    return FieldExtractionConfig(messages=messages, schema=schema)


def _validate_required_fields(
    fields: ExtractedFields,
    judge: Judge,
    scorer_name: str,
) -> None:
    """
    Validate that all required fields for a scorer are present.

    Args:
        fields: Extracted fields containing inputs, outputs, and expectations.
        judge: Judge instance to determine which fields are required.
        scorer_name: Name of the scorer for error messages.

    Raises:
        MlflowException: If any required fields are missing.
    """
    required_fields = {field.name for field in judge.get_input_fields()}
    missing_fields = []

    if "inputs" in required_fields and fields.inputs is None:
        missing_fields.append("inputs")
    if "outputs" in required_fields and fields.outputs is None:
        missing_fields.append("outputs")
    if "expectations" in required_fields and fields.expectations is None:
        missing_fields.append("expectations")

    if missing_fields:
        fields_str = ", ".join(missing_fields)
        raise MlflowException(
            f"{scorer_name} requires the following fields: {fields_str}. "
            "Provide them directly or pass a trace containing them."
        )


def resolve_scorer_fields(
    trace: Trace | None,
    judge: Judge,
    inputs: Any | None = None,
    outputs: Any | None = None,
    expectations: dict[str, Any] | None = None,
    model: str | None = None,
    extract_expectations: bool = False,
) -> ExtractedFields:
    """
    Resolve scorer fields from provided values or extract from trace if needed.

    Args:
        trace: MLflow trace object containing the execution to evaluate.
        judge: Judge instance to determine which fields need extraction.
        inputs: Input data to evaluate. If None, will be extracted from trace.
        outputs: Output data to evaluate. If None, will be extracted from trace.
        expectations: Dictionary of expected outcomes. If None, will be extracted from trace.
        model: Model URI to use for LLM-based extraction if needed.
        extract_expectations: If True, extract expectations from trace.

    Returns:
        ExtractedFields dataclass containing inputs, outputs, and expectations
    """
    if not trace:
        return ExtractedFields(inputs=inputs, outputs=outputs, expectations=expectations)

    inputs = resolve_inputs_from_trace(inputs, trace)
    outputs = resolve_outputs_from_trace(outputs, trace)
    if extract_expectations:
        expectations = resolve_expectations_from_trace(expectations, trace)

    input_field_names = {field.name for field in judge.get_input_fields()}
    needs_inputs = inputs is None and "inputs" in input_field_names
    needs_outputs = outputs is None and "outputs" in input_field_names

    if needs_inputs or needs_outputs:
        extraction_config = _construct_field_extraction_config(
            needs_inputs=needs_inputs,
            needs_outputs=needs_outputs,
        )

        try:
            extracted = get_chat_completions_with_structured_output(
                model_uri=model or get_default_model(),
                messages=extraction_config.messages,
                output_schema=extraction_config.schema,
                trace=trace,
            )
            if needs_inputs:
                inputs = inputs or extracted.inputs
            if needs_outputs:
                outputs = outputs or extracted.outputs
        except Exception as e:
            _logger.warning(
                "Failed to extract required fields from trace using LLM: %s",
                e,
            )

    return ExtractedFields(inputs=inputs, outputs=outputs, expectations=expectations)


def _sanitize_scorer_feedback(feedback: Feedback) -> Feedback:
    """Sanitize feedback values from LLM judges to ensure YES/NO consistency."""
    if feedback.value:
        if isinstance(feedback.value, CategoricalRating):
            return feedback

        if isinstance(feedback.value, str):
            value_str = feedback.value.strip().lower()

            if value_str in _AFFIRMATIVE_VALUES:
                feedback.value = CategoricalRating.YES
            elif value_str in _NEGATIVE_VALUES:
                feedback.value = CategoricalRating.NO
            else:
                feedback.value = CategoricalRating(value_str)

    return feedback


class BuiltInScorer(Judge):
    """
    Abstract base class for built-in scorers that share a common implementation.
    All built-in scorers should inherit from this class.
    """

    name: str
    required_columns: set[str] = set()

    @property
    @abstractmethod
    def instructions(self) -> str:
        """
        Get the instructions of what this scorer evaluates.
        """

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to handle builtin scorer serialization."""
        pydantic_model_data = pydantic.BaseModel.model_dump(self, mode="json", **kwargs)
        pydantic_model_data["instructions"] = self.instructions

        serialized = SerializedScorer(
            name=self.name,
            aggregations=self.aggregations,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            builtin_scorer_class=self.__class__.__name__,
            builtin_scorer_pydantic_data=pydantic_model_data,
        )

        return asdict(serialized)

    @classmethod
    def model_validate(cls, obj: SerializedScorer | dict[str, Any]) -> "BuiltInScorer":
        """Override model_validate to handle builtin scorer deserialization."""
        from mlflow.genai.scorers import builtin_scorers

        if isinstance(obj, SerializedScorer):
            serialized = obj
        else:
            if not isinstance(obj, dict) or "builtin_scorer_class" not in obj:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid builtin scorer data: expected a dictionary with "
                    f"'builtin_scorer_class' field, got {type(obj).__name__}."
                )

            try:
                serialized = SerializedScorer(**obj)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to parse serialized scorer data: {e}"
                )

        try:
            scorer_class = getattr(builtin_scorers, serialized.builtin_scorer_class)
        except AttributeError:
            raise MlflowException.invalid_parameter_value(
                f"Unknown builtin scorer class: {serialized.builtin_scorer_class}"
            )

        constructor_args = serialized.builtin_scorer_pydantic_data or {}

        return scorer_class(**constructor_args)

    def validate_columns(self, columns: set[str]) -> None:
        missing_columns = self.required_columns - columns
        if missing_columns:
            raise MissingColumnsException(self.name, missing_columns)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class RetrievalRelevance(BuiltInScorer):
    """
    Retrieval relevance measures whether each chunk is relevant to the input request.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "retrieval_relevance".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import RetrievalRelevance

        trace = mlflow.get_trace("<your-trace-id>")
        feedbacks = RetrievalRelevance(name="my_retrieval_relevance")(trace=trace)
        print(feedbacks)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[RetrievalRelevance()])
    """

    name: str = "retrieval_relevance"
    model: str | None = None
    required_columns: set[str] = {"inputs", "trace"}

    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return "Evaluates whether each retrieved context chunk is relevant to the input request."

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the RetrievalRelevance judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="trace",
                description=(
                    "The trace of the model's execution. Must contains at least one span with "
                    "type `RETRIEVER`. MLflow will extract the retrieved context from that span. "
                    "If multiple spans are found, MLflow will use the **last** one."
                ),
            ),
        ]

    def __call__(self, *, trace: Trace) -> Feedback:
        """
        Evaluate chunk relevance for each context chunk.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            A list of assessments evaluating the relevance of each context chunk.
            If the number of retrievers is N and each retriever has M chunks, the list will
            contain N * (M + 1) assessments. Each retriever span will emit M assessments
            for the relevance of its chunks and 1 assessment for the average relevance of all
            chunks.
        """
        request = extract_request_from_trace(trace)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedbacks.extend(self._compute_span_relevance(span_id, request, context))
        return feedbacks

    def _compute_span_relevance(
        self, span_id: str, request: str, chunks: list[dict[str, str]]
    ) -> list[Feedback]:
        """Compute the relevance of retrieved context for one retriever span."""
        from mlflow.genai.judges.prompts.retrieval_relevance import get_prompt

        model = self.model or get_default_model()

        chunk_feedbacks = []
        if model == "databricks":
            from databricks.agents.evals.judges import chunk_relevance

            chunk_feedbacks = chunk_relevance(
                request=request, retrieved_context=chunks, assessment_name=self.name
            )
        else:
            for chunk in chunks:
                prompt = get_prompt(request=request, context=chunk["content"])
                feedback = invoke_judge_model(model, prompt, assessment_name=self.name)
                sanitized_feedback = _sanitize_scorer_feedback(feedback)
                chunk_feedbacks.append(sanitized_feedback)

        for feedback in chunk_feedbacks:
            feedback.span_id = span_id

        if len(chunk_feedbacks) == 0:
            return []

        average = sum(f.value == "yes" for f in chunk_feedbacks) / len(chunk_feedbacks)

        span_level_feedback = Feedback(
            name=self.name + "/precision",
            value=average,
            source=chunk_feedbacks[0].source,
            span_id=span_id,
        )
        return [span_level_feedback] + chunk_feedbacks


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class RetrievalSufficiency(BuiltInScorer):
    """
    Retrieval sufficiency evaluates whether the retrieved documents provide all necessary
    information to generate the expected response.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "retrieval_sufficiency".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import RetrievalSufficiency

        trace = mlflow.get_trace("<your-trace-id>")
        feedback = RetrievalSufficiency(name="my_retrieval_sufficiency")(trace=trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[RetrievalSufficiency()])
    """

    name: str = "retrieval_sufficiency"
    model: str | None = None
    required_columns: set[str] = {"inputs", "trace"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return CONTEXT_SUFFICIENCY_PROMPT_INSTRUCTIONS

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the RetrievalSufficiency judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="trace",
                description=(
                    "The trace of the model's execution. Must contain at least one span with "
                    "type `RETRIEVER`. MLflow will extract the retrieved context from that span. "
                    "If multiple spans are found, MLflow will use the **last** one."
                ),
            ),
            JudgeField(
                name="expectations",
                description=(
                    "A dictionary of expectations for the response. This must contain either "
                    "`expected_response` or `expected_facts` key (optional)."
                ),
            ),
        ]

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name,
                ["expectations/expected_response or expectations/expected_facts"],
            )

    def __call__(
        self, *, trace: Trace, expectations: dict[str, Any] | None = None
    ) -> list[Feedback]:
        """
        Evaluate context sufficiency based on retrieved documents.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.
            expectations: A dictionary of expectations for the response. Either `expected_facts` or
                `expected_response` key is required. Alternatively, you can pass a trace annotated
                with `expected_facts` or `expected_response` label(s) and omit this argument.
        """
        request = extract_request_from_trace(trace)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        expectations = expectations or {}
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")
        if expected_facts is None or expected_response is None:
            for assessment in trace.info.assessments:
                if assessment.name == "expected_facts" and expected_facts is None:
                    expected_facts = assessment.value
                if assessment.name == "expected_response" and expected_response is None:
                    expected_response = assessment.value

        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedback = judges.is_context_sufficient(
                request=request,
                context=context,
                expected_response=expected_response,
                expected_facts=expected_facts,
                name=self.name,
                model=self.model,
            )
            feedback.span_id = span_id
            feedbacks.append(feedback)

        return feedbacks


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class RetrievalGroundedness(BuiltInScorer):
    """
    RetrievalGroundedness assesses whether the agent's response is aligned with the information
    provided in the retrieved context.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "retrieval_groundedness".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import RetrievalGroundedness

        trace = mlflow.get_trace("<your-trace-id>")
        feedback = RetrievalGroundedness(name="my_retrieval_groundedness")(trace=trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[RetrievalGroundedness()])
    """

    name: str = "retrieval_groundedness"
    model: str | None = None
    required_columns: set[str] = {"inputs", "trace"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return GROUNDEDNESS_PROMPT_INSTRUCTIONS

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the RetrievalGroundedness judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="trace",
                description=(
                    "The trace of the model's execution. Must contains at least one span with "
                    "type `RETRIEVER`. MLflow will extract the retrieved context from that span. "
                    "If multiple spans are found, MLflow will use the **last** one."
                ),
            ),
        ]

    def __call__(self, *, trace: Trace) -> list[Feedback]:
        """
        Evaluate groundedness of response against retrieved context.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the groundedness of the response.
        """
        request = extract_request_from_trace(trace)
        response = extract_response_from_trace(trace)
        span_id_to_context = extract_retrieval_context_from_trace(trace)
        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedback = judges.is_grounded(
                request=request,
                response=response,
                context=context,
                name=self.name,
                model=self.model,
            )
            feedback.span_id = span_id
            feedbacks.append(feedback)
        return feedbacks


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class Guidelines(BuiltInScorer):
    """
    Guideline adherence evaluates whether the agent's response follows specific constraints
    or instructions provided in the guidelines.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "guidelines".
        guidelines: A single guideline text or a list of guidelines.
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Guidelines

        english = Guidelines(
            name="english_guidelines",
            guidelines=["The response must be in English"],
        )
        feedback = english(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(feedback)

    Example (with evaluate):

    In the following example, the guidelines specified in the `english` and `clarify` scorers
    will be uniformly applied to all the examples in the dataset. The evaluation result will
    contains two scores "english" and "clarify".

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Guidelines

        english = Guidelines(
            name="english",
            guidelines=["The response must be in English"],
        )
        clarify = Guidelines(
            name="clarify",
            guidelines=["The response must be clear, coherent, and concise"],
        )

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            },
            {
                "inputs": {"question": "What is the capital of Germany?"},
                "outputs": "The capital of Germany is Berlin.",
            },
        ]
        mlflow.genai.evaluate(data=data, scorers=[english, clarify])
    """

    name: str = "guidelines"
    guidelines: str | list[str]
    model: str | None = None
    required_columns: set[str] = {"inputs", "outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return GUIDELINES_PROMPT_INSTRUCTIONS

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the Guidelines judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="inputs",
                description=(
                    "A dictionary of input data, e.g. "
                    "{'question': 'What is the capital of France?'}."
                ),
            ),
            JudgeField(
                name="outputs",
                description="The response from the model, e.g. 'The capital of France is Paris.'",
            ),
        ]

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate adherence to specified guidelines.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           and evaluate the inputs and outputs from the trace.
        2. Directly provide the inputs and outputs to evaluate.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
                Optional when trace is provided.
            outputs: The response from the model, e.g. "The capital of France is Paris."
                Optional when trace is provided.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                inputs and outputs will be automatically extracted from the trace.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        fields = resolve_scorer_fields(trace, self, inputs, outputs, model=self.model)
        _validate_required_fields(fields, self, "Guidelines scorer")

        feedback = judges.meets_guidelines(
            guidelines=self.guidelines,
            context={
                "request": parse_inputs_to_str(fields.inputs),
                "response": parse_outputs_to_str(fields.outputs),
            },
            name=self.name,
            model=self.model,
        )
        return _sanitize_scorer_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class ExpectationsGuidelines(BuiltInScorer):
    """
    This scorer evaluates whether the agent's response follows specific constraints
    or instructions provided for each row in the input dataset. This scorer is useful when
    you have a different set of guidelines for each example.

    To use this scorer, the input dataset should contain the `expectations` column with the
    `guidelines` field. Then pass this scorer to `mlflow.genai.evaluate` for running full
    evaluation on the input dataset.

    Args:
        name: The name of the scorer. Defaults to "expectations_guidelines".
        model: {{ model }}

    Example:

    In this example, the guidelines specified in the `guidelines` field of the `expectations`
    column will be applied to each example individually. The evaluation result will contain a
    single "expectations_guidelines" score.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import ExpectationsGuidelines

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "expectations": {
                    "guidelines": ["The response must be factual and concise"],
                },
            },
            {
                "inputs": {"question": "How to learn Python?"},
                "outputs": "You can read a book or take a course.",
                "expectations": {
                    "guidelines": ["The response must be helpful and encouraging"],
                },
            },
        ]
        mlflow.genai.evaluate(data=data, scorers=[ExpectationsGuidelines()])
    """

    name: str = "expectations_guidelines"
    model: str | None = None
    required_columns: set[str] = {"inputs", "outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return "Evaluates adherence to per-example guidelines provided in the expectations column."

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the ExpectationsGuidelines judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="inputs",
                description=(
                    "A dictionary of input data, e.g. "
                    "{'question': 'What is the capital of France?'}."
                ),
            ),
            JudgeField(
                name="outputs",
                description="The response from the model, e.g. 'The capital of France is Paris.'",
            ),
            JudgeField(
                name="expectations",
                description=(
                    "A dictionary containing guidelines for evaluation. "
                    "Must contain a 'guidelines' key (optional)."
                ),
            ),
        ]

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if "expectations/guidelines" not in columns:
            raise MissingColumnsException(self.name, ["expectations/guidelines"])

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate adherence to specified guidelines.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           and evaluate inputs, outputs, and expectations from the trace.
        2. Directly provide the inputs, outputs, and expectations to evaluate.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
                Optional when trace is provided.
            outputs: The response from the model, e.g. "The capital of France is Paris."
                Optional when trace is provided.
            expectations: A dictionary of expectations for the response. This must contain either
                `guidelines` key, which is used to evaluate the response against the guidelines
                specified in the `guidelines` field of the `expectations` column of the dataset.
                E.g., {"guidelines": ["The response must be factual and concise"]}
                Optional when trace is provided.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                missing inputs, outputs, and expectations will be automatically extracted from
                the trace.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        fields = resolve_scorer_fields(
            trace,
            self,
            inputs,
            outputs,
            expectations,
            model=self.model,
            extract_expectations=True,
        )
        _validate_required_fields(fields, self, "ExpectationsGuidelines scorer")

        guidelines = (fields.expectations or {}).get("guidelines")
        if not guidelines:
            raise MlflowException(
                "Guidelines must be specified in the `expectations` parameter or "
                "must be present in the trace."
            )
        feedback = judges.meets_guidelines(
            guidelines=guidelines,
            context={
                "request": parse_inputs_to_str(fields.inputs),
                "response": parse_outputs_to_str(fields.outputs),
            },
            name=self.name,
            model=self.model,
        )
        return _sanitize_scorer_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class RelevanceToQuery(BuiltInScorer):
    """
    Relevance ensures that the agent's response directly addresses the user's input without
    deviating into unrelated topics.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "relevance_to_query".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import RelevanceToQuery

        assessment = RelevanceToQuery(name="my_relevance_to_query")(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import RelevanceToQuery

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[RelevanceToQuery()])
    """

    name: str = "relevance_to_query"
    model: str | None = None
    required_columns: set[str] = {"inputs", "outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return RELEVANCE_TO_QUERY_PROMPT_INSTRUCTIONS

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the RelevanceToQuery judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="inputs",
                description=(
                    "A dictionary of input data, e.g. "
                    "{'question': 'What is the capital of France?'}."
                ),
            ),
            JudgeField(
                name="outputs",
                description="The response from the model, e.g. 'The capital of France is Paris.'",
            ),
        ]

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate relevance to the user's query.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           and evaluate the inputs and outputs from the trace.
        2. Directly provide the inputs and outputs to evaluate.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
                Optional when trace is provided.
            outputs: The response from the model, e.g. "The capital of France is Paris."
                Optional when trace is provided.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                inputs and outputs will be automatically extracted from the trace.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the relevance of the response to the query.
        """
        fields = resolve_scorer_fields(trace, self, inputs, outputs, model=self.model)
        _validate_required_fields(fields, self, "RelevanceToQuery scorer")

        # Use the existing scorer implementation with extracted/provided fields
        request = parse_inputs_to_str(fields.inputs)
        feedback = judges.is_context_relevant(
            request=request, context=fields.outputs, name=self.name, model=self.model
        )
        return _sanitize_scorer_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class Safety(BuiltInScorer):
    """
    Safety ensures that the agent's responses do not contain harmful, offensive, or toxic content.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "safety".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Safety

        assessment = Safety(name="my_safety")(outputs="The capital of France is Paris.")
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Safety

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[Safety()])
    """

    name: str = "safety"
    model: str | None = None
    required_columns: set[str] = {"inputs", "outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return "Ensures responses do not contain harmful, offensive, or toxic content."

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the Safety judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="outputs",
                description="The response from the model, e.g. 'The capital of France is Paris.'",
            ),
        ]

    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        *,
        outputs: Any | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate safety of the response.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           and evaluate the outputs from the trace.
        2. Directly provide the outputs to evaluate.

        Args:
            outputs: The response from the model, e.g. "The capital of France is Paris."
                Optional when trace is provided.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                outputs will be automatically extracted from the trace.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the safety of the response.
        """
        fields = resolve_scorer_fields(trace, self, outputs=outputs, model=self.model)
        _validate_required_fields(fields, self, "Safety scorer")

        feedback = judges.is_safe(
            content=parse_outputs_to_str(fields.outputs),
            name=self.name,
            model=self.model,
        )
        return _sanitize_scorer_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
class Correctness(BuiltInScorer):
    """
    Correctness ensures that the agent's responses are correct and accurate.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "correctness".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness

        assessment = Correctness(name="my_correctness")(
            inputs={
                "question": "What is the difference between reduceByKey and groupByKey in Spark?"
            },
            outputs=(
                "reduceByKey aggregates data before shuffling, whereas groupByKey "
                "shuffles all data, making reduceByKey more efficient."
            ),
            expectations=[
                {"expected_response": "reduceByKey aggregates data before shuffling"},
                {"expected_response": "groupByKey shuffles all data"},
            ],
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness

        data = [
            {
                "inputs": {
                    "question": (
                        "What is the difference between reduceByKey and groupByKey in Spark?"
                    )
                },
                "outputs": (
                    "reduceByKey aggregates data before shuffling, whereas groupByKey "
                    "shuffles all data, making reduceByKey more efficient."
                ),
                "expectations": {
                    "expected_response": (
                        "reduceByKey aggregates data before shuffling. "
                        "groupByKey shuffles all data"
                    ),
                },
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[Correctness()])
    """

    name: str = "correctness"
    model: str | None = None
    required_columns: set[str] = {"inputs", "outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return CORRECTNESS_PROMPT_INSTRUCTIONS

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name,
                ["expectations/expected_response or expectations/expected_facts"],
            )

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the Correctness judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="inputs",
                description=(
                    "A dictionary of input data, e.g. "
                    "{'question': 'What is the capital of France?'}."
                ),
            ),
            JudgeField(
                name="outputs",
                description="The response from the model, e.g. 'The capital of France is Paris.'",
            ),
            JudgeField(
                name="expectations",
                description=(
                    "A dictionary of expectations for the response. This must contain either "
                    "`expected_response` or `expected_facts` key, which is used to evaluate the "
                    "response against the expected response or facts respectively. "
                    "E.g., {'expected_facts': ['Paris', 'France', 'Capital']}"
                ),
            ),
        ]

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate correctness of the response against expectations.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           inputs, outputs, and expectations from the trace and its assessments.
        2. Directly provide inputs, outputs, and expectations to evaluate.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
                Optional when trace is provided.
            outputs: The response from the model, e.g. "The capital of France is Paris."
                Optional when trace is provided.
            expectations: A dictionary of expectations for the response. This must contain either
                `expected_response` or `expected_facts` key. Optional when trace is provided;
                will be extracted from trace's human assessment data if available.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                inputs, outputs, and expectations will be automatically extracted from the trace.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the correctness of the response.
        """
        fields = resolve_scorer_fields(
            trace,
            self,
            inputs,
            outputs,
            expectations,
            model=self.model,
            extract_expectations=True,
        )
        _validate_required_fields(fields, self, "Correctness scorer")

        if not fields.expectations or (
            fields.expectations.get("expected_response") is None
            and fields.expectations.get("expected_facts") is None
        ):
            raise MlflowException(
                "Correctness scorer requires either `expected_response` or `expected_facts` "
                "in the `expectations` dictionary."
            )

        request = parse_inputs_to_str(fields.inputs)
        response = parse_outputs_to_str(fields.outputs)
        expected_facts = fields.expectations.get("expected_facts")
        expected_response = fields.expectations.get("expected_response")

        feedback = judges.is_correct(
            request=request,
            response=response,
            expected_response=expected_response,
            expected_facts=expected_facts,
            name=self.name,
            model=self.model,
        )
        return _sanitize_scorer_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.5.0")
class Equivalence(BuiltInScorer):
    """
    Equivalence compares outputs against expected outputs for semantic equivalence.

    This scorer uses exact matching for numerical types (int, float, bool) and
    an LLM judge for text outputs to determine if they are semantically equivalent
    in both content and format.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` or `mlflow.genai.optimize_prompts` for evaluation.

    Args:
        name: The name of the scorer. Defaults to "equivalence".
        model: {{ model }}

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Equivalence

        # Numerical equivalence
        assessment = Equivalence()(
            outputs=42,
            expectations={"expected_response": 42},
        )
        print(assessment)  # value: ategoricalRating.YES, rationale: 'Exact numerical match'

        # Text equivalence
        assessment = Equivalence()(
            outputs="The capital is Paris",
            expectations={"expected_response": "Paris is the capital"},
        )
        print(assessment)  # value: CategoricalRating.YES (semantically equivalent)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Equivalence

        data = [
            {
                "outputs": "The capital is Paris",
                "expectations": {"expected_response": "Paris"},
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[Equivalence()])
    """

    name: str = "equivalence"
    model: str | None = None
    required_columns: set[str] = {"outputs"}

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return EQUIVALENCE_PROMPT_INSTRUCTIONS

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if "expectations/expected_response" not in columns:
            raise MissingColumnsException(self.name, {"expectations/expected_response"})

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the Equivalence scorer.

        Returns:
            List of JudgeField objects defining the input fields.
        """
        return [
            JudgeField(
                name="outputs",
                description="The actual output from the program to compare.",
            ),
            JudgeField(
                name="expectations",
                description=(
                    "A dictionary containing the expected output. Must contain an "
                    "'expected_response' key with the expected value, e.g. "
                    "{'expected_response': 'Paris'}."
                ),
            ),
        ]

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate output equivalence.

        This scorer can be used in two ways:
        1. Pass an MLflow trace object to automatically extract
           outputs and expectations from the trace and its assessments.
        2. Directly provide outputs and expectations to evaluate.

        Args:
            inputs: A dictionary of input data (optional, not used in evaluation).
            outputs: The actual output to compare. Optional when trace is provided.
            expectations: A dictionary containing the expected output. Must contain an
                'expected_response' key. Optional when trace is provided.
            trace: MLflow trace object containing the execution to evaluate. When provided,
                outputs and expectations will be automatically extracted from the trace.

        Returns:
            Feedback object with 'yes'/'no' value and rationale
        """
        from mlflow.genai.judges.builtin import _sanitize_feedback
        from mlflow.genai.judges.prompts.equivalence import (
            EQUIVALENCE_FEEDBACK_NAME,
            get_prompt,
        )

        # Use resolve_scorer_fields to extract fields from trace if provided
        fields = resolve_scorer_fields(
            trace,
            self,
            inputs,
            outputs,
            expectations,
            model=self.model,
            extract_expectations=True,
        )
        _validate_required_fields(fields, self, "Equivalence scorer")

        # Validate that expected_response is present
        if not fields.expectations or fields.expectations.get("expected_response") is None:
            raise MlflowException(
                "Equivalence scorer requires `expected_response` in the `expectations` dictionary."
            )

        # Extract the expected response
        expected_output = fields.expectations.get("expected_response")
        actual_output = fields.outputs

        # Handle exact match for numerical types
        if isinstance(actual_output, (int, float, bool)) and isinstance(
            expected_output, (int, float, bool)
        ):
            if math.isclose(actual_output, expected_output):
                return Feedback(
                    name=self.name,
                    value=CategoricalRating.YES,
                    rationale="Exact numerical match",
                )
            else:
                return Feedback(
                    name=self.name,
                    value=CategoricalRating.NO,
                    rationale=f"Values do not match: {actual_output} != {expected_output}",
                )

        # Convert to strings for comparison
        outputs_str = str(actual_output)
        expectations_str = str(expected_output)

        # Use exact match first
        if outputs_str == expectations_str:
            return Feedback(
                name=self.name,
                value=CategoricalRating.YES,
                rationale="Exact string match",
            )

        # Use LLM judge for semantic equivalence

        model = self.model or get_default_model()
        assessment_name = self.name or EQUIVALENCE_FEEDBACK_NAME

        prompt = get_prompt(
            output=outputs_str,
            expected_output=expectations_str,
        )
        feedback = invoke_judge_model(model, prompt, assessment_name=assessment_name)

        return _sanitize_feedback(feedback)


@experimental(version="3.0.0")
def get_all_scorers() -> list[BuiltInScorer]:
    """
    Returns a list of all built-in scorers.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import get_all_scorers

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "expectations": {"expected_response": "Paris is the capital city of France."},
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=get_all_scorers())
    """
    scorers = [
        ExpectationsGuidelines(),
        Correctness(),
        RelevanceToQuery(),
        RetrievalSufficiency(),
        RetrievalGroundedness(),
        Equivalence(),
    ]
    if is_databricks_uri(mlflow.get_tracking_uri()):
        scorers.extend([Safety(), RetrievalRelevance()])
    return scorers


class MissingColumnsException(MlflowException):
    def __init__(self, scorer: str, missing_columns: set[str]):
        self.scorer = scorer
        self.missing_columns = list(missing_columns)
        super().__init__(
            f"The following columns are required for the scorer {scorer}: {missing_columns}"
        )
