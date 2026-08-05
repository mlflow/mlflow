from __future__ import annotations

from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    from mlflow.genai.scorers.base import Scorer


class Preset:
    """A named, immutable collection of scorers for common evaluation patterns.

    A preset is NOT a Scorer subclass — it is a grouping mechanism that gets
    flattened into individual scorers at validation time.

    Args:
        name: The name of the preset.
        scorers: A list of scorer instances to include.

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import Preset, Safety, Fluency

        my_preset = Preset("my_eval", scorers=[Safety(), Fluency()])

        # Use directly in evaluate
        mlflow.genai.evaluate(data=data, scorers=[my_preset])

        # Or expand manually
        mlflow.genai.evaluate(data=data, scorers=my_preset.scorers)
    """

    def __init__(self, name: str, scorers: list["Scorer"]):
        from mlflow.genai.scorers.base import Scorer

        if not isinstance(name, str) or not name:
            raise MlflowException.invalid_parameter_value(
                "Preset `name` must be a non-empty string."
            )

        if not isinstance(scorers, list) or len(scorers) == 0:
            raise MlflowException.invalid_parameter_value(
                "Preset `scorers` must be a non-empty list of scorer instances."
            )

        for s in scorers:
            if not isinstance(s, Scorer):
                raise MlflowException.invalid_parameter_value(
                    f"All items in `scorers` must be Scorer instances, got {type(s).__name__}."
                )

        seen = set()
        for s in scorers:
            key = (type(s), s.name)
            if key in seen:
                raise MlflowException.invalid_parameter_value(
                    f"Duplicate scorer in preset: {s.name!r} (type {type(s).__name__}). "
                    f"If you need two scorers of the same type, give them different names."
                )
            seen.add(key)

        self._name = name
        self._scorers = list(scorers)

    @property
    def name(self) -> str:
        return self._name

    @property
    def scorers(self) -> list["Scorer"]:
        return list(self._scorers)

    def register(self, *, experiment_id: str | None = None):
        raise NotImplementedError(
            "Preset persistence is not yet implemented. This will be available in a future release."
        )

    def copy(self, *, to_experiment_id: str):
        raise NotImplementedError(
            "Preset persistence is not yet implemented. This will be available in a future release."
        )

    def __iter__(self):
        return iter(self._scorers)

    def __len__(self):
        return len(self._scorers)

    def __repr__(self):
        scorer_names = [s.name for s in self._scorers]
        return f"Preset(name={self._name!r}, scorers={scorer_names})"


class Rag(Preset):
    """Built-in preset for evaluating retrieval-augmented generation pipelines.

    Includes: RetrievalRelevance, RetrievalGroundedness, RelevanceToQuery,
    Safety, Completeness.

    Args:
        model: Optional judge model URI to use for all scorers
            (e.g. ``"openai:/gpt-4o"``).

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import Rag

        mlflow.genai.evaluate(data=data, scorers=[Rag()])
    """

    def __init__(self, *, model: str | None = None):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            RelevanceToQuery,
            RetrievalGroundedness,
            RetrievalRelevance,
            Safety,
        )

        scorers = [
            RetrievalRelevance(model=model) if model else RetrievalRelevance(),
            RetrievalGroundedness(model=model) if model else RetrievalGroundedness(),
            RelevanceToQuery(model=model) if model else RelevanceToQuery(),
            Safety(model=model) if model else Safety(),
            Completeness(model=model) if model else Completeness(),
        ]
        super().__init__(name="rag", scorers=scorers)


class Agent(Preset):
    """Built-in preset for evaluating single-turn tool-calling agents.

    Includes: ToolCallCorrectness, ToolCallEfficiency, RelevanceToQuery,
    Safety, Completeness.

    Args:
        model: Optional judge model URI to use for all scorers
            (e.g. ``"openai:/gpt-4o"``).

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import Agent

        mlflow.genai.evaluate(data=data, scorers=[Agent()])
    """

    def __init__(self, *, model: str | None = None):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            RelevanceToQuery,
            Safety,
            ToolCallCorrectness,
            ToolCallEfficiency,
        )

        scorers = [
            ToolCallCorrectness(model=model) if model else ToolCallCorrectness(),
            ToolCallEfficiency(model=model) if model else ToolCallEfficiency(),
            RelevanceToQuery(model=model) if model else RelevanceToQuery(),
            Safety(model=model) if model else Safety(),
            Completeness(model=model) if model else Completeness(),
        ]
        super().__init__(name="agent", scorers=scorers)


class ConversationalAgent(Preset):
    """Built-in preset for evaluating multi-turn conversational agents.

    Includes all Agent scorers plus: UserFrustration,
    ConversationCompleteness, ConversationalSafety,
    ConversationalToolCallEfficiency, KnowledgeRetention.

    Args:
        model: Optional judge model URI to use for all scorers
            (e.g. ``"openai:/gpt-4o"``).

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import ConversationalAgent

        mlflow.genai.evaluate(data=data, scorers=[ConversationalAgent()])
    """

    def __init__(self, *, model: str | None = None):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            ConversationalSafety,
            ConversationalToolCallEfficiency,
            ConversationCompleteness,
            KnowledgeRetention,
            RelevanceToQuery,
            Safety,
            ToolCallCorrectness,
            ToolCallEfficiency,
            UserFrustration,
        )

        scorers = [
            ToolCallCorrectness(model=model) if model else ToolCallCorrectness(),
            ToolCallEfficiency(model=model) if model else ToolCallEfficiency(),
            RelevanceToQuery(model=model) if model else RelevanceToQuery(),
            Safety(model=model) if model else Safety(),
            Completeness(model=model) if model else Completeness(),
            UserFrustration(model=model) if model else UserFrustration(),
            ConversationCompleteness(model=model) if model else ConversationCompleteness(),
            ConversationalSafety(model=model) if model else ConversationalSafety(),
            ConversationalToolCallEfficiency(model=model)
            if model
            else ConversationalToolCallEfficiency(),
            KnowledgeRetention(model=model) if model else KnowledgeRetention(),
        ]
        super().__init__(name="conversational_agent", scorers=scorers)
