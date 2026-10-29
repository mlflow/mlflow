from types import SimpleNamespace

import pytest

from mlflow.entities.issue import IssueCategory
from mlflow.telemetry.issue_detection_utils import (
    count_issues_by_category,
    normalize_issue_categories,
    normalize_issue_category_counts,
)


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        (None, None),
        ([], []),
        ([IssueCategory.CORRECTNESS, "cost"], ["correctness", "cost"]),
        (["private category", "another private", "cost", "cost"], ["Custom", "cost"]),
    ],
)
def test_normalize_issue_categories(categories, expected):
    assert normalize_issue_categories(categories) == expected


def test_count_each_issue_once_per_normalized_category():
    issues = [
        SimpleNamespace(categories=["correctness", "private", "private two"]),
        SimpleNamespace(categories=["correctness", "correctness", "cost"]),
        SimpleNamespace(categories=None),
    ]
    assert count_issues_by_category(issues) == {"correctness": 2, "Custom": 1, "cost": 1}


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (None, None),
        ({}, {}),
        ({"safety": 2, "private": 1, "private two": 1}, {"safety": 2, "Custom": 2}),
        ({"correctness": 0, "invalid": "private text", "safety": -1}, {"correctness": 0}),
    ],
)
def test_normalize_issue_category_counts(counts, expected):
    assert normalize_issue_category_counts(counts) == expected
