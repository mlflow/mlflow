"""
Online scorer entities and configuration.

This module contains entities for online scorer configuration used by the store layer
and online scoring infrastructure.
"""

from dataclasses import dataclass


@dataclass
class OnlineScorer:
    """
    Internal entity representing a serialized scorer and its online execution configuration.

    This entity specifies how a scorer should be applied to traces in an online/real-time
    manner, including which traces to score (via filter_string) and at what sampling rate.
    The scorer itself is stored in serialized form for execution by the online scoring jobs.
    """

    name: str
    experiment_id: str
    serialized_scorer: str
    sample_rate: float
    filter_string: str | None = None


@dataclass
class OnlineScoringConfig:
    """
    Internal entity representing the online configuration for a scorer.

    This configuration controls how a scorer is applied to traces in an online/real-time
    manner. It defines sampling rates and optional filters for selecting which traces
    should be scored.
    """

    online_scoring_config_id: str
    scorer_id: str
    sample_rate: float
    filter_string: str | None = None

    def to_dict(self) -> dict[str, str | float]:
        """Convert the entity to a dictionary for JSON serialization."""
        result: dict[str, str | float] = {
            "online_scoring_config_id": self.online_scoring_config_id,
            "scorer_id": self.scorer_id,
            "sample_rate": self.sample_rate,
        }
        if self.filter_string is not None:
            result["filter_string"] = self.filter_string
        return result
