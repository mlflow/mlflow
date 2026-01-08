"""Conversational metrics for evaluating multi-turn dialogue performance."""

from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.deepeval import DeepEvalScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class TurnRelevancy(DeepEvalScorer):
    """
    Evaluates the relevance of each conversation turn.

    This multi-turn metric assesses whether each response in a conversation is relevant
    to the corresponding user query. It evaluates coherence across the entire dialogue.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import TurnRelevancy

            scorer = TurnRelevancy(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])  # List of conversation turns

    """

    metric_name: ClassVar[str] = "TurnRelevancy"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class RoleAdherence(DeepEvalScorer):
    """
    Evaluates whether the agent stays in character throughout the conversation.

    This multi-turn metric assesses if the agent consistently maintains its assigned
    role, personality, and behavioral constraints across all conversation turns.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import RoleAdherence

            scorer = RoleAdherence(threshold=0.8)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "RoleAdherence"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class KnowledgeRetention(DeepEvalScorer):
    """
    Evaluates the chatbot's ability to retain and use information from earlier in the conversation.

    This multi-turn metric assesses whether the agent remembers and appropriately
    references information from previous turns in the conversation, demonstrating
    context awareness.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import KnowledgeRetention

            scorer = KnowledgeRetention(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "KnowledgeRetention"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ConversationCompleteness(DeepEvalScorer):
    """
    Evaluates whether the conversation satisfies the user's needs and goals.

    This multi-turn metric assesses if the conversation reaches a satisfactory conclusion,
    addressing all aspects of the user's original request or question.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import (
                ConversationCompleteness,
            )

            scorer = ConversationCompleteness(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "ConversationCompleteness"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class GoalAccuracy(DeepEvalScorer):
    """
    Evaluates the accuracy of achieving conversation goals in a multi-turn context.

    This multi-turn metric assesses whether the agent successfully achieves the
    specified goals or objectives throughout the conversation, measuring goal-oriented
    effectiveness.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import GoalAccuracy

            scorer = GoalAccuracy(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "GoalAccuracy"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ToolUse(DeepEvalScorer):
    """
    Evaluates the effectiveness of tool usage throughout a conversation.

    This multi-turn metric assesses whether the agent appropriately uses available
    tools across multiple conversation turns, measuring tool selection and usage
    effectiveness in a dialogue context.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ToolUse

            scorer = ToolUse(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "ToolUse"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class TopicAdherence(DeepEvalScorer):
    """
    Evaluates adherence to specified topics throughout a conversation.

    This multi-turn metric assesses whether the agent stays on topic across the
    entire conversation, avoiding unnecessary digressions or topic drift.

    Note: This is a multi-turn metric that requires a list of traces representing
    conversation turns.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import TopicAdherence

            scorer = TopicAdherence(threshold=0.7)
            feedback = scorer(traces=[trace1, trace2, trace3])

    """

    metric_name: ClassVar[str] = "TopicAdherence"
