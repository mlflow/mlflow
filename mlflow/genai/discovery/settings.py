"""The detection priorities an experiment has saved, read server-side.

The Issues tab writes these as a single JSON experiment tag and sends them with every manual run,
so until now nothing on the server had to read the tag. A scheduled run has no browser to send
them, so it reads the tag itself and interprets it the way the tab does -- including the two
defaulting rules below -- so a schedule and a manual run analyze an experiment the same way.

Only the category selection is read here, because that is all a scheduled run can act on today:
the free-text priorities reach the pipeline through an argument the manual path supplies, and
wiring them in is a separate change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from mlflow.entities.experiment import Experiment
from mlflow.genai.discovery.constants import DEFAULT_CATEGORIES
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_PRIORITIES

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionSettings:
    """
    What an experiment has asked detection to focus on.

    Args:
        categories: The categories a run should look for, already defaulted -- never empty.
    """

    categories: tuple[str, ...]


def default_detection_settings() -> DetectionSettings:
    return DetectionSettings(categories=tuple(DEFAULT_CATEGORIES))


def parse_detection_settings(raw_value: str | None) -> DetectionSettings:
    """
    Parse the saved priorities, falling back to the defaults for anything unreadable.

    Two defaulting rules are kept the same as the tab's, so a schedule analyzes an experiment the
    way its owner sees it configured:

    - A built-in category the tag does not mention is treated as selected. It is a category added
      after the owner last saved, not a decision they made.
    - A selection that comes out empty means every category, which is what the tab runs when
      nothing is ticked.
    """
    if not raw_value:
        return default_detection_settings()
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        _logger.warning(
            "Ignoring unreadable issue-detection priorities; the run will use every category"
        )
        return default_detection_settings()
    if not isinstance(parsed, dict):
        return default_detection_settings()

    raw_categories = parsed.get("categories")
    if not isinstance(raw_categories, dict):
        return default_detection_settings()
    selected = dict.fromkeys(DEFAULT_CATEGORIES, True)
    for category, is_selected in raw_categories.items():
        if isinstance(category, str) and category.strip() and isinstance(is_selected, bool):
            selected[category.strip()] = is_selected
    categories = tuple(category for category, is_selected in selected.items() if is_selected)
    return DetectionSettings(categories=categories or tuple(DEFAULT_CATEGORIES))


def read_detection_settings(experiment: Experiment) -> DetectionSettings:
    return parse_detection_settings(experiment.tags.get(MLFLOW_ISSUE_DETECTION_PRIORITIES))
