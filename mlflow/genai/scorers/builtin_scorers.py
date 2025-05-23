from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional

from mlflow.entities import Assessment
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import BuiltInScorer
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.utils.annotations import experimental

GENAI_CONFIG_NAME = "databricks-agent"


class _BaseBuiltInScorer(BuiltInScorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    required_columns: set[str] = set()

    # Avoid direct mutation of built-in scorer fields like name. Overriding the default setting must
    # be done through the `with_config` method. This is because the short-hand syntax like
    # `mlflow.genai.scorers.chunk_relevance` returns a single instance rather than a new instance.
    def __setattr__(self, name: str, value: Any) -> None:
        raise MlflowException(
            "Built-in scorer fields are immutable. Please use the `with_config` method to "
            "get a new instance with the custom field values.",
            error_code=BAD_REQUEST,
        )

    def __call__(self, **kwargs):
        try:
            from databricks.agents.evals import judges
        except ImportError:
            raise ImportError(
                "databricks-agents is not installed. Please install it with "
                "`pip install databricks-agents`"
            )

        if self.name and self.name in set(dir(judges)):
            import pandas as pd

            from mlflow.genai.evaluation.utils import _convert_to_legacy_eval_set

            converted_kwargs = _convert_to_legacy_eval_set(pd.DataFrame([kwargs])).iloc[0].to_dict()
            return getattr(judges, self.name)(**converted_kwargs)
        elif self.name:
            raise ValueError(
                f"The scorer '{self.name}' doesn't currently have a usable implementation in the "
                "databricks-agents package."
            )
        else:
            raise ValueError("This scorer isn't recognized since it doesn't have a name.")

    @abstractmethod
    def with_config(self, **kwargs) -> "_BaseBuiltInScorer":
        """
        Get a new scorer instance with the given configuration, such as name, global guidelines.

        Override this method with the appropriate config keys. This method must return the scorer
        instance itself with the updated configuration.
        """

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = deepcopy(evaluation_config)
        metrics = config.setdefault(GENAI_CONFIG_NAME, {}).setdefault("metrics", [])
        if self.name not in metrics:
            metrics.append(self.name)
        return config

    def validate_columns(self, columns: set[str]) -> None:
        missing_columns = self.required_columns - columns
        if missing_columns:
            raise MissingColumnsException(self.name, missing_columns)


# === Builtin Scorers ===
@experimental
class ChunkRelevance(_BaseBuiltInScorer):
    """
    Chunk relevance measures whether each chunk is relevant to the input request.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.chunk_relevance` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import chunk_relevance

        assessment = chunk_relevance(
            inputs={"question": "What is the capital of France?"},
            retrieved_context=[
                {"content": "Paris is the capital city of France."},
                {"content": "The chicken crossed the road."},
            ],
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "retrieved_context": [
                    {"content": "Paris is the capital city of France."},
                    {"content": "The chicken crossed the road."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[chunk_relevance])
    """

    name: str = "chunk_relevance"
    required_columns: set[str] = {"inputs", "retrieved_context"}

    def __call__(self, *, inputs: Any, retrieved_context: list[dict[str, Any]]) -> list[Assessment]:
        """
        Evaluate chunk relevance for each context chunk.

        Args:
            inputs: The input data.
            retrieved_context: The retrieved context.

        Returns:
            A list of assessments evaluating the relevance of each context chunk.

        Example:

        .. code-block:: python

            from mlflow.genai.scorers import chunk_relevance

            assessments = chunk_relevance(
                inputs={"question": "What is the capital of France?"},
                retrieved_context=[{"content": "Paris is the capital city of France."}],
            )
        """
        return super().__call__(inputs=inputs, retrieved_context=retrieved_context)

    def with_config(self, *, name: str = "chunk_relevance") -> "ChunkRelevance":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "chunk_relevance".
        """
        return ChunkRelevance(name=name)


@experimental
class ContextSufficiency(_BaseBuiltInScorer):
    """
    Context sufficiency evaluates whether the retrieved documents provide all necessary
    information to generate the expected response.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.context_sufficiency` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import context_sufficiency

        assessment = context_sufficiency(
            inputs={"question": "What is the capital of France?"},
            retrieved_context=[{"content": "Paris is the capital city of France."}],
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "retrieved_context": [{"content": "Paris is the capital city of France."}],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[context_sufficiency])
    """

    name: str = "context_sufficiency"
    required_columns: set[str] = {"inputs", "retrieved_context", "expectations/expected_response"}

    def __call__(self, *, inputs: Any, retrieved_context: list[dict[str, Any]]) -> Assessment:
        """Evaluate context sufficiency based on retrieved documents."""
        return super().__call__(inputs=inputs, retrieved_context=retrieved_context)

    def with_config(self, *, name: str = "context_sufficiency") -> "ContextSufficiency":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "context_sufficiency".
        """
        return ContextSufficiency(name=name)


@experimental
class Groundedness(_BaseBuiltInScorer):
    """
    Groundedness assesses whether the agent's response is aligned with the information provided
    in the retrieved context.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.groundedness` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import groundedness

        assessment = groundedness(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
            retrieved_context=[{"content": "Paris is the capital city of France."}],
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "retrieved_context": [{"content": "Paris is the capital city of France."}],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[groundedness])
    """

    name: str = "groundedness"
    required_columns: set[str] = {"inputs", "outputs", "retrieved_context"}

    def __call__(
        self, *, inputs: Any, outputs: Any, retrieved_context: list[dict[str, Any]]
    ) -> Assessment:
        """Evaluate groundedness of response against context."""
        return super().__call__(inputs=inputs, outputs=outputs, retrieved_context=retrieved_context)

    def with_config(self, *, name: str = "groundedness") -> "Groundedness":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "groundedness".

        Returns:
            The updated Groundedness scorer instance.
        """
        return Groundedness(name=name)


@experimental
class GuidelineAdherence(_BaseBuiltInScorer):
    """
    Guideline adherence evaluates whether the agent's response follows specific constraints
    or instructions provided in the guidelines.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.guideline_adherence` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    There are two different ways to specify judges, depending on the use case:

    **1. Global Guidelines**

    If you want to evaluate all the response with a single set of guidelines, you can specify
    the guidelines in the `guidelines` parameter of this scorer.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        # Create a global judge
        english = guideline_adherence.with_config(
            name="english_guidelines",
            global_guidelines=["The response must be in English"],
        )
        assessment = english(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):

    In the following example, the guidelines specified in the `english` and `clarify` scorers
    will be uniformly applied to all the examples in the dataset. The evaluation result will
    contains two scores "english" and "clarify".

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        english = guideline_adherence.with_config(
            name="english",
            global_guidelines=["The response must be in English"],
        )
        clarify = guideline_adherence.with_config(
            name="clarify",
            global_guidelines=["The response must be clear, coherent, and concise"],
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

    **2. Per-Example Guidelines**

    When you have a different set of guidelines for each example, you can specify the guidelines
    in the `guidelines` field of the `expectations` column of the input dataset. Alternatively,
    you can annotate a trace with "guidelines" expectation and use the trace as an input data.

    Example:

    In this example, the guidelines specified in the `guidelines` field of the `expectations`
    column will be applied to each example individually. The evaluation result will contain a
    single "guideline_adherence" score.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

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
        mlflow.genai.evaluate(data=data, scorers=[guideline_adherence])
    """

    name: str = "guideline_adherence"
    global_guidelines: Optional[list[str]] = None
    required_columns: set[str] = {"inputs", "outputs"}

    def update_evaluation_config(self, evaluation_config) -> dict:
        # Metric name should always be "guideline_adherence" regardless of the custom name
        config = deepcopy(evaluation_config)
        metrics = config.setdefault(GENAI_CONFIG_NAME, {}).setdefault("metrics", [])
        if "guideline_adherence" not in metrics:
            metrics.append("guideline_adherence")

        # If global guidelines are specified, add it to the config
        if self.global_guidelines:
            # NB: The agent eval harness will take multiple global guidelines in a dictionary format
            #   where the key is the name of the global guideline judge. Therefore, when multiple
            #   judges are specified, we merge them into a single dictionary.
            #   https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#examples
            global_guidelines = config[GENAI_CONFIG_NAME].get("global_guidelines", {})
            global_guidelines[self.name] = self.global_guidelines
            config[GENAI_CONFIG_NAME]["global_guidelines"] = global_guidelines

        return config

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        # If no global guidelines are specified, the guidelines must exist in the input dataset
        if not self.global_guidelines and "expectations/guidelines" not in columns:
            raise MissingColumnsException(self.name, ["expectations/guidelines"])

    def __call__(
        self,
        *,
        inputs: Any,
        outputs: Any,
        guidelines: dict[str, list[str]],
        guidelines_context: dict[str, Any],
    ) -> Assessment:
        """Evaluate adherence to specified guidelines."""
        return super().__call__(
            inputs=inputs,
            outputs=outputs,
            guidelines=guidelines,
            guidelines_context=guidelines_context,
        )

    def with_config(
        self,
        *,
        name: str = "guideline_adherence",
        global_guidelines: Optional[list[str]] = None,
    ) -> "GuidelineAdherence":
        """
        Get a new scorer instance with the given name and global guidelines.

        Args:
            name: The name of the scorer. Default is "guideline_adherence".
            global_guidelines: A list of global guidelines to be used for evaluation.
                If not provided, the scorer will use the per-row guidelines in the input dataset.

        Returns:
            The updated GuidelineAdherence scorer instance.

        Example:

        .. code-block:: python

            from mlflow.genai.scorers import guideline_adherence

            is_english = guideline_adherence.with_config(
                name="is_english", global_guidelines=["The response must be in English"]
            )

            mlflow.genai.evaluate(data=data, scorers=[is_english])
        """
        return GuidelineAdherence(name=name, global_guidelines=global_guidelines)


@experimental
class RelevanceToQuery(_BaseBuiltInScorer):
    """
    Relevance ensures that the agent's response directly addresses the user's input without
    deviating into unrelated topics.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.relevance_to_query` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import relevance_to_query

        assessment = relevance_to_query(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import relevance_to_query

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[relevance_to_query])
    """

    name: str = "relevance_to_query"
    required_columns: set[str] = {"inputs", "outputs"}

    def __call__(self, *, inputs: Any, outputs: Any) -> Assessment:
        """Evaluate relevance to the user's query."""
        return super().__call__(inputs=inputs, outputs=outputs)

    def with_config(self, *, name: str = "relevance_to_query") -> "RelevanceToQuery":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "relevance_to_query".
        """
        return RelevanceToQuery(name=name)


@experimental
class Safety(_BaseBuiltInScorer):
    """
    Safety ensures that the agent's responses do not contain harmful, offensive, or toxic content.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.safety` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import safety

        assessment = safety(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import safety

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[safety])
    """

    name: str = "safety"
    required_columns: set[str] = {"inputs", "outputs"}

    def __call__(self, *, inputs: Any, outputs: Any) -> Assessment:
        """Evaluate safety of the response."""
        return super().__call__(inputs=inputs, outputs=outputs)

    def with_config(self, *, name: str = "safety") -> "Safety":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "safety".
        """
        return Safety(name=name)


@experimental
class Correctness(_BaseBuiltInScorer):
    """
    Correctness ensures that the agent's responses are correct and accurate.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.correctness` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import correctness

        assessment = correctness(
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
        from mlflow.genai.scorers import correctness

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
                "expectations": [
                    {"expected_response": "reduceByKey aggregates data before shuffling"},
                    {"expected_response": "groupByKey shuffles all data"},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[correctness])
    """

    name: str = "correctness"
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

    def __call__(self, *, inputs: Any, outputs: Any, expectations: list[str]) -> Assessment:
        """Evaluate correctness of the response against expectations."""
        return super().__call__(inputs=inputs, outputs=outputs, expectations=expectations)

    def with_config(self, *, name: str = "correctness") -> "Correctness":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The new name of the scorer. Default is "correctness".
        """
        return Correctness(name=name)


# === Shorthand for getting builtin scorer instances ===
chunk_relevance = ChunkRelevance()
context_sufficiency = ContextSufficiency()
groundedness = Groundedness()
guideline_adherence = GuidelineAdherence()
relevance_to_query = RelevanceToQuery()
safety = Safety()
correctness = Correctness()


# === Shorthand for all builtin RAG scorers ===
@experimental
def get_rag_scorers() -> list[BuiltInScorer]:
    """
    Returns a list of built-in scorers for evaluating RAG models. Contains scorers
    chunk_relevance, context_sufficiency, groundedness, and relevance_to_query.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import get_rag_scorers

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "retrieved_context": [
                    {"content": "Paris is the capital city of France."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=get_rag_scorers())
    """
    return [
        chunk_relevance,
        context_sufficiency,
        groundedness,
        relevance_to_query,
    ]


@experimental
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
                "retrieved_context": [
                    {"content": "Paris is the capital city of France."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=get_all_scorers())
    """
    return get_rag_scorers() + [
        guideline_adherence,
        safety,
        correctness,
    ]


class MissingColumnsException(MlflowException):
    def __init__(self, scorer: str, missing_columns: set[str]):
        self.scorer = scorer
        self.missing_columns = list(missing_columns)
        super().__init__(
            f"The following columns are required for the scorer {scorer}: {missing_columns}"
        )
