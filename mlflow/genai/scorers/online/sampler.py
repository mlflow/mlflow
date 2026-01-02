"""Dense sampling strategy for online scoring."""

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from mlflow.genai.scorers.base import Scorer

if TYPE_CHECKING:
    from mlflow.genai.scorers.online.processor import OnlineScorer

_logger = logging.getLogger(__name__)


class OnlineScorerSampler:
    """
    Samples scorers for traces using dense sampling strategy.

    Dense sampling ensures traces that are selected get thorough coverage:
    - Sort scorers by sample_rate descending
    - Use conditional probability: if a scorer is rejected, skip all lower-rate scorers
    """

    def __init__(self, online_scorers: list["OnlineScorer"]):
        self._online_scorers = online_scorers
        self._sample_rates: dict[str, float] = {}
        self._scorers: dict[str, Scorer] = {}
        for online_scorer in online_scorers:
            scorer_dict = json.loads(online_scorer.serialized_scorer)
            scorer = Scorer.model_validate(scorer_dict)
            self._sample_rates[scorer.name] = online_scorer.sample_rate
            self._scorers[scorer.name] = scorer

    def get_filter_strings(self) -> set[str | None]:
        """Get all unique filter strings from online scorers."""
        return {s.filter_string for s in self._online_scorers}

    def get_scorers_for_filter(
        self, filter_string: str | None, session_level: bool
    ) -> list[Scorer]:
        """Get scorers matching the filter string and session level."""
        result = []
        for online_scorer in self._online_scorers:
            scorer_dict = json.loads(online_scorer.serialized_scorer)
            scorer = self._scorers.get(scorer_dict.get("name"))
            if (
                scorer
                and online_scorer.filter_string == filter_string
                and scorer.is_session_level_scorer == session_level
            ):
                result.append(scorer)
        return result

    def sample(self, entity_id: str, scorers: list[Scorer]) -> list[Scorer]:
        """
        Apply dense sampling to select scorers for an entity.

        Dense sampling ensures selected entities receive comprehensive evaluation across
        multiple scorers, rather than spreading scorers thinly across all entities.
        For example, with two scorers at 50% and 25% sample rates:
        - 50% of entities get both scorers (dense coverage)
        - 25% get only the first scorer
        - 25% get no scorers
        This enables better comparisons between scorers on the same entities.

        Args:
            entity_id: The trace ID or session ID to sample for.
            scorers: List of scorers to sample from.

        Returns:
            A subset of scorers selected via conditional probability waterfall.
        """
        if not scorers:
            return []

        # Sort by sample rate descending
        sorted_scorers = sorted(
            scorers,
            key=lambda s: self._sample_rates.get(s.name, 1.0),
            reverse=True,
        )

        selected = []
        prev_rate = 1.0

        for scorer in sorted_scorers:
            rate = self._sample_rates.get(scorer.name, 1.0)
            conditional_rate = rate / prev_rate if prev_rate > 0 else 0

            # Hash entity_id + scorer name to get deterministic value in [0, 1]
            hash_input = f"{entity_id}:{scorer.name}"
            hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) / (2**256)

            if hash_value > conditional_rate:
                break

            selected.append(scorer)
            prev_rate = rate

        _logger.debug(
            f"Sampled {len(selected)}/{len(scorers)} scorers for entity {entity_id[:8]}..."
        )
        return selected
