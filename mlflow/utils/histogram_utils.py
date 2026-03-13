import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class HistogramData:
    """
    Represents histogram data for a single step.

    Args:
        name: Histogram name/key
        step: Training step/iteration
        timestamp: Time when histogram was recorded (milliseconds since epoch)
        bin_edges: Array of bin edges (length n+1 for n bins)
        counts: Array of counts per bin (length n)
        min_value: Minimum value in the data (optional)
        max_value: Maximum value in the data (optional)
    """

    def __init__(
        self,
        name: str,
        step: int,
        timestamp: int,
        bin_edges: list[float] | np.ndarray,
        counts: list[float] | np.ndarray,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        self.name = name
        self.step = step
        self.timestamp = timestamp
        self.bin_edges = np.asarray(bin_edges, dtype=np.float64).tolist()
        self.counts = np.asarray(counts, dtype=np.float64).tolist()
        self.min_value = min_value
        self.max_value = max_value

        # Validation
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError(
                f"bin_edges must have length n+1 where n is the number of bins. "
                f"Got bin_edges length {len(self.bin_edges)} and counts length {len(self.counts)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "step": self.step,
            "timestamp": self.timestamp,
            "bin_edges": self.bin_edges,
            "counts": self.counts,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistogramData":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            step=data["step"],
            timestamp=data["timestamp"],
            bin_edges=data["bin_edges"],
            counts=data["counts"],
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
        )


def compute_histogram_from_values(
    values: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute histogram from raw values.

    Args:
        values: Raw data values as numpy array
        num_bins: Number of bins to use

    Returns:
        Tuple of (bin_edges, counts, min_value, max_value)
    """
    # Remove NaN values
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        _logger.warning("All values are NaN, creating empty histogram")
        return np.array([0.0, 1.0]), np.array([0.0]), 0.0, 0.0

    min_val = float(np.min(valid_values))
    max_val = float(np.max(valid_values))

    # Handle case where all values are the same
    if min_val == max_val:
        bin_edges = np.array([min_val - 0.5, max_val + 0.5])
        counts = np.array([float(len(valid_values))])
    else:
        counts, bin_edges = np.histogram(valid_values, bins=num_bins)
        counts = counts.astype(np.float64)

    return bin_edges, counts, min_val, max_val


def save_histogram_to_json(histogram: HistogramData, file_path: str | Path) -> None:
    """
    Save a single histogram to a JSON file.

    Args:
        histogram: HistogramData instance
        file_path: Path to JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(histogram.to_dict(), f)


def load_histogram_from_json(file_path: str | Path) -> HistogramData:
    """
    Load a single histogram from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        HistogramData instance
    """
    with open(file_path) as f:
        return HistogramData.from_dict(json.load(f))
