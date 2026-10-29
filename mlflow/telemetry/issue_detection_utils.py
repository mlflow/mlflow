import os
from collections import Counter
from typing import Literal

from mlflow.entities.issue import Issue, IssueCategory

_BUILTIN_CATEGORIES = frozenset(category.value for category in IssueCategory)


def get_issue_detection_trigger() -> Literal["sdk", "adhoc", "schedule"]:
    match os.environ.get("_MLFLOW_SERVER_JOB_NAME"):
        case "invoke_issue_detection":
            return "adhoc"
        case "scheduled_issue_detection":
            return "schedule"
        case _:
            return "sdk"


def normalize_issue_categories(categories: list[str] | None) -> list[str] | None:
    if categories is None:
        return None
    return list(
        dict.fromkeys(
            str(category) if category in _BUILTIN_CATEGORIES else "Custom"
            for category in categories
        )
    )


def count_issues_by_category(issues: list[Issue]) -> dict[str, int]:
    return dict(
        Counter(
            category
            for issue in issues
            for category in normalize_issue_categories(issue.categories) or []
        )
    )
