"""
Online scorer entities and configuration.

This module contains entities for online scorer configuration used by the store layer
and online scoring infrastructure.
"""

from dataclasses import dataclass


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
    experiment_id: str
    filter_string: str | None = None

    def to_dict(self) -> dict[str, str | float]:
        result: dict[str, str | float] = {
            "online_scoring_config_id": self.online_scoring_config_id,
            "scorer_id": self.scorer_id,
            "sample_rate": self.sample_rate,
            "experiment_id": self.experiment_id,
        }
        if self.filter_string is not None:
            result["filter_string"] = self.filter_string
        return result


@dataclass
class OnlineScorer:
    """
    Entity representing an active online scorer with its configuration.

    Used by the online scoring infrastructure to apply scorers to traces.
    """

    name: str
    experiment_id: str
    serialized_scorer: str
    sample_rate: float
    filter_string: str | None = None
