import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/verify.yml"
HEAD = "a" * 40
BASE = "b" * 40
MEDIA = "https://github.com/user-attachments/assets/example"


@dataclass
class WorkflowRunner:
    directory: Path
    env: dict[str, str]

    def run(self, job: str, name: str, **env: str) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text())
        step = next(step for step in workflow["jobs"][job]["steps"] if step.get("name") == name)
        subprocess.run(
            ["bash", "-eo", "pipefail", "-c", step["run"]],
            env=self.env | env,
            check=True,
        )

    def posts(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob("post*.json"))]


@pytest.fixture
def runner(tmp_path: Path) -> WorkflowRunner:
    gh = tmp_path / "gh"
    gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["MOCK_ROOT"])
with (root / "gh.log").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if any("/collaborators/" in arg for arg in args):
    print(os.environ.get("MOCK_ROLE", "maintain"))
elif any("reviews?per_page=" in arg for arg in args):
    if os.environ.get("MOCK_ACCEPT_FIRST") == "true" and (root / "post1.json").exists():
        print(json.dumps([[{"user": {"login": "github-actions[bot]"},
                            "body": "<!-- mlflow-verify-run:10-1 -->",
                            "html_url": "https://example.test/review"}]]))
    else:
        print("[[]]")
elif any(arg.endswith("/reviews") for arg in args):
    count = len(list(root.glob("post*.json"))) + 1
    payload = Path(args[args.index("--input") + 1]).read_text()
    (root / f"post{count}.json").write_text(payload)
    if count == 1 and os.environ.get("MOCK_FAIL_FIRST") == "true":
        print("HTTP 422", file=sys.stderr)
        sys.exit(1)
    print("https://example.test/review")
else:
    print(json.dumps({"state": "open", "number": 123, "title": "Fix", "body": "PR description",
                      "html_url": "https://example.test/pr", "draft": False,
                      "head": {"sha": os.environ["MOCK_HEAD"]},
                      "base": {"sha": os.environ["MOCK_BASE"],
                               "ref": os.environ.get("MOCK_BASE_REF", "master")}}))
"""
    )
    gh.chmod(0o755)
    env = dict(os.environ) | {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "MOCK_ROOT": str(tmp_path),
        "MOCK_HEAD": HEAD,
        "MOCK_BASE": BASE,
        "GH_TOKEN": "unused",
        "REPO": "mlflow/mlflow",
        "PR_NUMBER": "123",
        "COMMENTER": "alice",
        "HEAD_SHA": HEAD,
        "BASE_REF": "master",
        "RUN_URL": "https://example.test/run",
        "GITHUB_RUN_ID": "10",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "PAYLOAD_PATH": str(tmp_path / "payload.json"),
        "REVIEW_READY": "true",
    }
    return WorkflowRunner(tmp_path, env)


def test_skill_requires_explicit_invocation() -> None:
    skill = (ROOT / ".claude/skills/verify/SKILL.md").read_text()
    frontmatter = yaml.safe_load(skill.split("---", 2)[1])
    policy = yaml.safe_load((ROOT / ".claude/skills/verify/agents/openai.yaml").read_text())
    assert frontmatter["disable-model-invocation"] is True
    assert policy["policy"]["allow_implicit_invocation"] is False


def test_workflow_separates_trusted_and_pr_checkouts() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["verify"]["steps"]
    trusted = next(step for step in steps if step.get("name") == "Checkout trusted workflow files")
    pr = next(step for step in steps if step.get("name") == "Checkout PR head")
    assert trusted["with"]["path"] == "trusted"
    assert trusted["with"]["ref"] == "${{ github.sha }}"
    assert pr["with"]["path"] == "pr"
    assert trusted["with"]["persist-credentials"] is False
    assert pr["with"]["persist-credentials"] is False


@pytest.mark.parametrize(
    ("event", "comment", "valid"),
    [
        ("APPROVE", {"path": "f", "line": 1, "side": "RIGHT", "body": "Evidence"}, True),
        (
            "COMMENT",
            {"path": "f", "line": 1, "side": "LEFT", "body": "🔴 **CRITICAL:** Broken"},
            True,
        ),
        (
            "APPROVE",
            {"path": "f", "line": 1, "side": "RIGHT", "body": "🔴 **CRITICAL:** Broken"},
            False,
        ),
        (
            "COMMENT",
            {"path": "f", "line": 2, "side": "RIGHT", "start_line": 1, "body": "Evidence"},
            False,
        ),
    ],
)
def test_schema_enforces_review_event_and_inline_ranges(
    tmp_path: Path, event: str, comment: dict[str, Any], valid: bool
) -> None:
    schema = ROOT / ".claude/skills/verify/payload.schema.json"
    payload = {"event": event, "body": "Verification results", "comments": [comment]}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    result = subprocess.run(
        ["check-jsonschema", "--schemafile", schema, path], capture_output=True, text=True
    )
    assert (result.returncode == 0) is valid, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("comment", "role", "called_gh"),
    [("/verify-more", "maintain", False), ("/verify", "write", True)],
)
def test_gate_rejects_noncommands_and_nonmaintainers(
    runner: WorkflowRunner, comment: str, role: str, called_gh: bool
) -> None:
    runner.run("gate", "Check maintainer and snapshot PR", COMMENT=comment, MOCK_ROLE=role)
    assert not (runner.directory / "output").exists()
    assert (runner.directory / "gh.log").exists() is called_gh


def test_gate_pins_context_and_keeps_maintainer_instructions(runner: WorkflowRunner) -> None:
    runner.run("gate", "Check maintainer and snapshot PR", COMMENT="/verify\nCheck the UI")
    assert "allowed=true" in (runner.directory / "output").read_text()
    context = json.loads((runner.directory / "verify-context/pr.json").read_text())
    assert context["head"] == HEAD
    assert context["base"] == BASE
    assert context["base_ref"] == "master"
    assert context["instructions"] == "\nCheck the UI"


@pytest.mark.parametrize(("event", "side"), [("APPROVE", "RIGHT"), ("COMMENT", "LEFT")])
def test_review_preserves_event_and_inline_location(
    runner: WorkflowRunner, event: str, side: str
) -> None:
    comment = {
        "path": "mlflow/ui.py",
        "line": 3,
        "side": side,
        "start_line": 1,
        "start_side": side,
        "body": f"![UI]({MEDIA})",
    }
    (runner.directory / "payload.json").write_text(
        json.dumps({
            "event": event,
            "body": "Observed the behavior.",
            "commit_id": "c" * 40,
            "comments": [comment],
        })
    )
    runner.run("post_review", "Publish review")
    [review] = runner.posts()
    assert review["event"] == event
    assert review["commit_id"] == HEAD
    assert review["comments"] == [comment]
    assert "<!-- mlflow-verify-run:10-1 -->" in review["body"]


def test_incomplete_verification_posts_a_comment(runner: WorkflowRunner) -> None:
    runner.run("post_review", "Publish review", REVIEW_READY="false")
    [review] = runner.posts()
    assert review["event"] == "COMMENT"
    assert "did not complete" in review["body"]


def test_dispatch_preview_escapes_review_without_posting(runner: WorkflowRunner) -> None:
    (runner.directory / "payload.json").write_text(
        json.dumps({"event": "COMMENT", "body": "<unsafe>", "comments": []})
    )
    runner.run("post_review", "Preview review")
    summary = (runner.directory / "summary").read_text()
    assert "<pre>" in summary
    assert "&lt;unsafe&gt;" in summary
    assert runner.posts() == []


@pytest.mark.parametrize("change", [{"MOCK_HEAD": "c" * 40}, {"MOCK_BASE_REF": "other"}])
def test_changed_pr_is_not_reviewed(runner: WorkflowRunner, change: dict[str, str]) -> None:
    runner.run("post_review", "Publish review", **change)
    assert runner.posts() == []


@pytest.mark.parametrize("accepted", [False, True])
def test_failed_submission_does_not_lose_findings_or_duplicate_a_review(
    runner: WorkflowRunner, accepted: bool
) -> None:
    (runner.directory / "payload.json").write_text(
        json.dumps({
            "event": "COMMENT",
            "body": "The behavior failed.",
            "comments": [
                {"path": "mlflow/ui.py", "line": 3, "side": "RIGHT", "body": f"![UI]({MEDIA})"}
            ],
        })
    )
    runner.run(
        "post_review",
        "Publish review",
        MOCK_FAIL_FIRST="true",
        MOCK_ACCEPT_FIRST=str(accepted).lower(),
    )
    if accepted:
        assert len(runner.posts()) == 1
    else:
        [_, fallback] = runner.posts()
        assert fallback["event"] == "COMMENT"
        assert "comments" not in fallback
        assert "mlflow/ui.py:3" in fallback["body"]
        assert MEDIA in fallback["body"]
