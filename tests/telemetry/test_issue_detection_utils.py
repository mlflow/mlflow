from types import SimpleNamespace

import pytest

from mlflow.entities.issue import IssueCategory
from mlflow.telemetry.issue_detection_utils import (
    count_issues_by_category,
    get_issue_detection_trigger,
    normalize_issue_categories,
)


@pytest.mark.parametrize(
    ("job_name", "expected"),
    [
        (None, "sdk"),
        ("another_job", "sdk"),
        ("invoke_issue_detection", "adhoc"),
        ("scheduled_issue_detection", "schedule"),
    ],
)
def test_get_issue_detection_trigger(monkeypatch, job_name, expected):
    if job_name is None:
        monkeypatch.delenv("_MLFLOW_SERVER_JOB_NAME", raising=False)
    else:
        monkeypatch.setenv("_MLFLOW_SERVER_JOB_NAME", job_name)
    assert get_issue_detection_trigger() == expected


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
