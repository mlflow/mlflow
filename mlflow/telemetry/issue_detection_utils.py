from collections import Counter

from mlflow.entities.issue import Issue, IssueCategory

_BUILTIN_CATEGORIES = frozenset(category.value for category in IssueCategory)


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


def normalize_issue_category_counts(counts: dict[str, int] | None) -> dict[str, int] | None:
    if not isinstance(counts, dict):
        return None
    normalized = Counter()
    for category, count in counts.items():
        if type(count) is int and count >= 0:
            category = str(category) if category in _BUILTIN_CATEGORIES else "Custom"
            normalized[category] += count
    return dict(normalized)
