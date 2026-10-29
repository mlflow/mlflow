import subprocess
import sys
import textwrap
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
        ("another_job", "unknown"),
        ("", "unknown"),
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
        (["Correctness", " cost ", "SAFETY"], ["correctness", "cost", "safety"]),
        (["Correctness", " correctness ", "CORRECTNESS"], ["correctness"]),
        ([" Private category ", "PRIVATE CATEGORY"], ["Custom"]),
        (["private category", "another private", "cost", "cost"], ["Custom", "cost"]),
    ],
)
def test_normalize_issue_categories(categories, expected):
    assert normalize_issue_categories(categories) == expected


def test_count_each_issue_once_per_normalized_category():
    issues = [
        SimpleNamespace(categories=[" Correctness ", "private", "private two"]),
        SimpleNamespace(categories=["correctness", "CORRECTNESS", " Cost "]),
        SimpleNamespace(categories=None),
    ]
    assert count_issues_by_category(issues) == {"correctness": 2, "Custom": 1, "cost": 1}


def test_issue_detection_telemetry_does_not_import_server_dependencies():
    subprocess.check_call([
        sys.executable,
        "-c",
        textwrap.dedent("""
                import importlib.abc
                import sys

                class BlockServerImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split(".")[0] in {"flask", "huey"} or (
                            fullname == "mlflow.server" or fullname.startswith("mlflow.server.")
                        ):
                            raise AssertionError(f"Unexpected server dependency: {fullname}")

                sys.meta_path.insert(0, BlockServerImports())

                from mlflow.environment_variables import (
                    _MLFLOW_SERVER_JOB_ID,
                    _MLFLOW_SERVER_JOB_NAME,
                )
                from mlflow.telemetry.events import DiscoverIssuesEvent

                _MLFLOW_SERVER_JOB_NAME.unset()
                assert DiscoverIssuesEvent.parse({})["trigger"] == "sdk"
                _MLFLOW_SERVER_JOB_NAME.set("invoke_issue_detection")
                _MLFLOW_SERVER_JOB_ID.set("job-123")
                params = DiscoverIssuesEvent.parse({"categories": [" Correctness ", "private"]})
                assert params["trigger"] == "adhoc"
                assert params["job_id"] == "job-123"
                assert params["categories"] == ["correctness", "Custom"]
            """),
    ])
