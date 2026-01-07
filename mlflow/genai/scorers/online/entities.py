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
    Internal entity representing a serialized scorer and its online execution configuration.

    This entity combines the scorer's executable form (name and serialized_scorer) with
    its configuration (OnlineScoringConfig) that specifies how it should be applied to
    traces in an online/real-time manner.
    """

    name: str
    serialized_scorer: str
    online_config: OnlineScoringConfig


@dataclass
class CompletedSession:
    """
    Metadata about a session that has been determined complete and is eligible for online scoring.

    Contains only the session ID and timestamp range, not the actual trace data.
    """

    session_id: str
    first_trace_timestamp_ms: int
    last_trace_timestamp_ms: int
