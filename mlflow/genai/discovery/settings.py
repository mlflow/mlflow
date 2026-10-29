"""The detection priorities an experiment has saved, read server-side.

The Issues tab writes these as a single JSON experiment tag and sends them with every manual run,
so until now nothing on the server had to read the tag. A scheduled run has no browser to send
them, so it reads the tag itself and interprets it the way the tab does -- including the two
defaulting rules below -- so a schedule and a manual run analyze an experiment the same way.

Both halves of the tag are read: the category selection and the owner's free-text priorities,
which reach the pipeline through the same argument a manual run supplies.
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
        instructions: The owner's free-text detection priorities, or ``""`` when none are saved.
    """

    categories: tuple[str, ...]
    instructions: str = ""


def default_detection_settings() -> DetectionSettings:
    return DetectionSettings(categories=tuple(DEFAULT_CATEGORIES))


def parse_detection_settings(raw_value: str | None) -> DetectionSettings:
    """
    Parse the saved priorities, falling back to the defaults for anything unreadable.

    Read exactly the way the tab reads it, so a schedule analyzes an experiment the way its owner
    sees it configured:

    - The selected categories are the keys the tag sets to ``true``. A category the tag does not
      mention is not selected, and neither is one whose value is not a boolean, since neither is
      a selection the owner made.
    - A selection that comes out empty means every category, which is what the tab runs when
      nothing is ticked. Restricting a run to no categories at all would let it report nothing.

    The free-text priorities are passed on as they were written. They are capped and wrapped in
    their delimited block by the prompt builder, the same way a manual run's are, so there is
    nothing to validate here that is not validated there.
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

    instructions = parsed.get("instructions")
    instructions = instructions if isinstance(instructions, str) else ""
    raw_categories = parsed.get("categories")
    if not isinstance(raw_categories, dict):
        return DetectionSettings(categories=tuple(DEFAULT_CATEGORIES), instructions=instructions)
    selected: dict[str, bool] = {}
    for category, is_selected in raw_categories.items():
        if isinstance(category, str) and category.strip() and isinstance(is_selected, bool):
            # Last write wins, as it does for the object the tab parses, so two keys differing
            # only in surrounding whitespace cannot produce the category twice.
            selected[category.strip()] = is_selected
    categories = tuple(category for category, is_selected in selected.items() if is_selected)
    return DetectionSettings(
        categories=categories or tuple(DEFAULT_CATEGORIES), instructions=instructions
    )


def read_detection_settings(experiment: Experiment) -> DetectionSettings:
    return parse_detection_settings(experiment.tags.get(MLFLOW_ISSUE_DETECTION_PRIORITIES))
