from dataclasses import asdict
from typing import Any

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import judges
from mlflow.genai.judges.builtin import _MODEL_API_DOC, requires_databricks_agents
from mlflow.genai.scorers.base import _SERIALIZATION_VERSION, Scorer, ScorerKind, SerializedScorer
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring
from mlflow.utils.uri import is_databricks_uri

GENAI_CONFIG_NAME = "databricks-agent"


def _validate_tracking_uri_is_databricks(scorer_name: str) -> None:
    """Validate that the current tracking URI is set to Databricks.

    Args:
        scorer_name: The name of the scorer being validated (for error messages).

    Raises:
        MlflowException: If the MLflow tracking URI is not set to Databricks.
    """
    from mlflow.utils.uri import is_databricks_uri

    if not is_databricks_uri(mlflow.get_tracking_uri()):
        raise MlflowException(
            f"The {scorer_name} scorer is only available in Databricks managed "
            "MLflow. If you have a Databricks workspace, please set MLflow tracking "
            "URI to the workspace by calling `mlflow.set_tracking_uri('databricks')`."
        )


class BuiltInScorer(Scorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    name: str
    required_columns: set[str] = set()

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to handle builtin scorer serialization."""
        # Use mode='json' to automatically convert sets to lists for JSON compatibility
        from pydantic import BaseModel

        pydantic_model_data = BaseModel.model_dump(self, mode="json", **kwargs)

        # Create serialized scorer with core fields
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
    def model_validate(cls, obj) -> "BuiltInScorer":
        """Override model_validate to handle builtin scorer deserialization."""
        if not isinstance(obj, dict) or "builtin_scorer_class" not in obj:
            raise MlflowException.invalid_parameter_value(
                f"Invalid builtin scorer data: expected a dictionary with 'builtin_scorer_class'"
                f" field, got {type(obj).__name__}."
            )

        from mlflow.genai.scorers import builtin_scorers
        from mlflow.genai.scorers.base import SerializedScorer

        # Parse the serialized data using our dataclass
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

        # Use the builtin_scorer_pydantic_data directly to reconstruct the scorer
        constructor_args = serialized.builtin_scorer_pydantic_data or {}

        return scorer_class(**constructor_args)

    def validate_columns(self, columns: set[str]) -> None:
        missing_columns = self.required_columns - columns
        if missing_columns:
            raise MissingColumnsException(self.name, missing_columns)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN


# === Builtin Scorers ===
@experimental(version="3.0.0")
class RetrievalRelevance(BuiltInScorer):
    """
    Retrieval relevance measures whether each chunk is relevant to the input request.

    .. warning::
        This scorer is currently only available in Databricks managed MLflow. It requires
        the `databricks-agents` package and will only work when the MLflow tracking URI
        is set to Databricks.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

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
    required_columns: set[str] = {"inputs", "trace"}

    def __init__(self, /, **kwargs):
        _validate_tracking_uri_is_databricks("RetrievalRelevance")
        super().__init__(**kwargs)

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
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedbacks.extend(self._compute_span_relevance(span_id, request, context))
        return feedbacks

    @requires_databricks_agents
    def _compute_span_relevance(
        self, span_id: str, request: str, chunks: dict[str, str]
    ) -> list[Feedback]:
        """Compute the relevance of retrieved context for one retriever span."""
        from databricks.agents.evals.judges import chunk_relevance

        # Compute relevance for each chunk. Call `chunk_relevance` judge directly
        # to get a list of feedbacks with ids.
        chunk_feedbacks = chunk_relevance(
            request=request, retrieved_context=chunks, assessment_name=self.name
        )
        for feedback in chunk_feedbacks:
            feedback.span_id = span_id

        if len(chunk_feedbacks) == 0:
            return []

        # Compute average relevance across all chunks.
        # NB: Handling error feedback as 0.0 relevance for simplicity.
        average = sum(f.value == "yes" for f in chunk_feedbacks) / len(chunk_feedbacks)

        span_level_feedback = Feedback(
            # NB: Adding a special suffix for span-level aggregation so that UI can distinguish
            # it from the chunk-level score and render it on span correctly.
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

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name, ["expectations/expected_response or expectations/expected_facts"]
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
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        # If expectations are explicitly provided, use them.
        expectations = expectations or {}
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")

        # As a fallback, use the trace annotations as expectations.
        if expected_facts is None or expected_response is None:
            for assessment in trace.info.assessments:
                if assessment.name == "expected_facts" and expected_facts is None:
                    expected_facts = assessment.value
                if assessment.name == "expected_response" and expected_response is None:
                    expected_response = assessment.value

        # This scorer returns a list of feedbacks, one for retriever span in the trace.
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
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        response = parse_outputs_to_str(trace.data.spans[0].outputs)
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

        # Create a global judge
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

    def __call__(
        self,
        *,
        inputs: dict[str, Any],
        outputs: Any,
    ) -> Feedback:
        """
        Evaluate adherence to specified guidelines.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        return judges.meets_guidelines(
            guidelines=self.guidelines,
            context={
                "request": parse_inputs_to_str(inputs),
                "response": parse_outputs_to_str(outputs),
            },
            name=self.name,
            model=self.model,
        )


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

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if "expectations/guidelines" not in columns:
            raise MissingColumnsException(self.name, ["expectations/guidelines"])

    def __call__(
        self,
        *,
        inputs: dict[str, Any],
        outputs: Any,
        expectations: dict[str, Any] | None = None,
    ) -> Feedback:
        """
        Evaluate adherence to specified guidelines.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."
            expectations: A dictionary of expectations for the response. This must contain either
                `guidelines` key, which is used to evaluate the response against the guidelines
                specified in the `guidelines` field of the `expectations` column of the dataset.
                E.g., {"guidelines": ["The response must be factual and concise"]}

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        guidelines = (expectations or {}).get("guidelines")
        if not guidelines:
            raise MlflowException(
                "Guidelines must be specified in the `expectations` parameter or "
                "must be present in the trace."
            )

        return judges.meets_guidelines(
            guidelines=guidelines,
            context={
                "request": parse_inputs_to_str(inputs),
                "response": parse_outputs_to_str(outputs),
            },
            name=self.name,
            model=self.model,
        )


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

    def __call__(self, *, inputs: dict[str, Any], outputs: Any) -> Feedback:
        """
        Evaluate relevance to the user's query.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the relevance of the response to the query.
        """
        request = parse_inputs_to_str(inputs)
        # NB: Reuse is_context_relevant judge to evaluate response
        return judges.is_context_relevant(
            request=request, context=outputs, name=self.name, model=self.model
        )


@experimental(version="3.0.0")
class Safety(BuiltInScorer):
    """
    Safety ensures that the agent's responses do not contain harmful, offensive, or toxic content.

    .. warning::
        This scorer is currently only available in Databricks managed MLflow. It requires
        the `databricks-agents` package and will only work when the MLflow tracking URI
        is set to Databricks.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

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
    required_columns: set[str] = {"inputs", "outputs"}

    def __init__(self, /, **kwargs):
        _validate_tracking_uri_is_databricks("Safety")
        super().__init__(**kwargs)

    def __call__(self, *, outputs: Any) -> Feedback:
        """
        Evaluate safety of the response.

        Args:
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the safety of the response.
        """
        return judges.is_safe(content=parse_outputs_to_str(outputs), name=self.name)


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

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name, ["expectations/expected_response or expectations/expected_facts"]
            )

    def __call__(
        self, *, inputs: dict[str, Any], outputs: Any, expectations: dict[str, Any]
    ) -> Feedback:
        """
        Evaluate correctness of the response against expectations.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."
            expectations: A dictionary of expectations for the response. This must contain either
                `expected_response` or `expected_facts` key, which is used to evaluate the response
                against the expected response or facts respectively.
                E.g., {"expected_facts": ["Paris", "France", "Capital"]}

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the correctness of the response.
        """
        request = parse_inputs_to_str(inputs)
        response = parse_outputs_to_str(outputs)
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")

        if expected_response is None and expected_facts is None:
            raise MlflowException(
                "Correctness scorer requires either `expected_response` or `expected_facts` "
                "in the `expectations` dictionary."
            )

        return judges.is_correct(
            request=request,
            response=response,
            expected_response=expected_response,
            expected_facts=expected_facts,
            name=self.name,
            model=self.model,
        )


# === Shorthand for getting preset of builtin scorers ===
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
    ]
    # TODO: Open-source these two scorers
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
