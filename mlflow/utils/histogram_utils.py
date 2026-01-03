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
    values: np.ndarray | list,
    num_bins: int = 30,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute histogram from raw values.

    Args:
        values: Raw data values
        num_bins: Number of bins to use

    Returns:
        Tuple of (bin_edges, counts, min_value, max_value)
    """
    values = np.asarray(values, dtype=np.float64)

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
    Save histogram data to JSON file.

    Args:
        histogram: HistogramData instance
        file_path: Path to save JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(histogram.to_dict(), f, indent=2)


def load_histograms_from_json(file_path: str | Path) -> list[HistogramData]:
    """
    Load histogram data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of HistogramData instances
    """
    with open(file_path) as f:
        data = json.load(f)

    # Support both single histogram and list of histograms
    if isinstance(data, dict):
        return [HistogramData.from_dict(data)]
    elif isinstance(data, list):
        return [HistogramData.from_dict(h) for h in data]
    else:
        raise ValueError(f"Invalid histogram data format: {type(data)}")


def append_histogram_to_json(histogram: HistogramData, file_path: str | Path) -> None:
    """
    Append histogram data to existing JSON file or create new one.

    Args:
        histogram: HistogramData instance
        file_path: Path to JSON file
    """
    file_path = Path(file_path)

    # Load existing histograms if file exists
    if file_path.exists():
        try:
            histograms = load_histograms_from_json(file_path)
        except Exception as e:
            _logger.warning(f"Failed to load existing histograms from {file_path}: {e}")
            histograms = []
    else:
        histograms = []

    # Append new histogram
    histograms.append(histogram)

    # Save all histograms
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump([h.to_dict() for h in histograms], f, indent=2)
