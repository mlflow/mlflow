from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class TopicAdherence(RagasScorer):
    """
    Evaluates whether the AI system adheres to specified topics during interaction.

    This metric assesses if the agent stays on topic and avoids answering queries
    outside its designated domain of interest.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import TopicAdherence

            scorer = TopicAdherence()
            feedback = scorer(
                trace=trace,
                expectations={
                    "reference_topics": ["machine learning", "data science"],
                },
            )

            # or for sessions:
            session = mlflow.search_traces(
                filter_string="request_metadata.mlflow.trace.session='{session_id}'",
                return_type="list",
            )
            feedback = scorer(
                session=session,
                expectations={
                    "reference_topics": ["machine learning", "data science"],
                },
            )
    """

    metric_name: ClassVar[str] = "TopicAdherence"


@experimental(version="3.9.0")
class ToolCallAccuracy(RagasScorer):
    """
    Evaluates the accuracy of tool calls made by an agent.

    This deterministic metric compares the actual tool calls made by the agent
    against expected tool calls, considering both the tool names and their
    arguments. It can evaluate in strict order or flexible order mode.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ToolCallAccuracy

            scorer = ToolCallAccuracy()
            feedback = scorer(
                trace=trace,
                expectations={
                    "expected_tool_calls": [
                        {"name": "weather_check", "arguments": {"location": "Paris"}},
                        {"name": "uv_index_lookup", "arguments": {"location": "Paris"}},
                    ]
                },
            )

            # or for sessions:
            session = mlflow.search_traces(
                filter_string="request_metadata.mlflow.trace.session='{session_id}'",
                return_type="list",
            )
            feedback = scorer(
                session=session,
                expectations={
                    "expected_tool_calls": [
                        {"name": "weather_check", "arguments": {"location": "Paris"}},
                        {"name": "uv_index_lookup", "arguments": {"location": "Paris"}},
                    ]
                },
            )
    """

    metric_name: ClassVar[str] = "ToolCallAccuracy"


@experimental(version="3.9.0")
class ToolCallF1(RagasScorer):
    """
    Calculates F1 score between expected and actual tool calls.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ToolCallF1

            scorer = ToolCallF1()
            feedback = scorer(
                trace=trace,
                expectations={
                    "expected_tool_calls": [
                        {"name": "weather_check", "arguments": {"location": "Paris"}},
                    ]
                },
            )

            # or for sessions:
            session = mlflow.search_traces(
                filter_string="request_metadata.mlflow.trace.session='{session_id}'",
                return_type="list",
            )
            feedback = scorer(
                session=session,
                expectations={
                    "expected_tool_calls": [
                        {"name": "weather_check", "arguments": {"location": "Paris"}},
                    ]
                },
            )
    """

    metric_name: ClassVar[str] = "ToolCallF1"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class AgentGoalAccuracyWithReference(RagasScorer):
    """
    Evaluates whether the agent achieved the user's goal compared to the expectations.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import AgentGoalAccuracyWithReference

            scorer = AgentGoalAccuracyWithReference(model="openai:/gpt-4")
            feedback = scorer(
                trace=trace,
                expectations={"expected_output": "Table booked at a Chinese restaurant for 8pm"},
            )
            # or for sessions:
            session = mlflow.search_traces(
                filter_string="request_metadata.mlflow.trace.session='{session_id}'",
                return_type="list",
            )
            feedback = scorer(
                session=session,
                expectations={"expected_output": "Table booked at a Chinese restaurant for 8pm"},
            )
    """

    metric_name: ClassVar[str] = "AgentGoalAccuracyWithReference"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class AgentGoalAccuracyWithoutReference(RagasScorer):
    """
    Evaluates whether the agent achieved the user's goal without expectations.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import AgentGoalAccuracyWithoutReference

            scorer = AgentGoalAccuracyWithoutReference(model="openai:/gpt-4")
            feedback = scorer(trace=trace)

            # or for sessions:
            session = mlflow.search_traces(
                filter_string="request_metadata.mlflow.trace.session='{session_id}'",
                return_type="list",
            )
            feedback = scorer(session=session)
    """

    metric_name: ClassVar[str] = "AgentGoalAccuracyWithoutReference"
