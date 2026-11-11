import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def test_nailaopus_review_request_survives_publication_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "resolve_workflow", Path(__file__).resolve().parents[1] / "resolve/scripts/workflow.py"
    )
    assert spec is not None
    assert spec.loader is not None
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("DEFAULT_BRANCH", "master")
    monkeypatch.setenv("PR_CHECKOUT", str(tmp_path))
    workflow.git("init", "--quiet")
    (tmp_path / "fix.py").write_text("fixed\n")
    workflow.git("add", "fix.py")
    workflow.git(
        "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fix"
    )
    head = workflow.git("rev-parse", "HEAD")
    monkeypatch.setattr(workflow, "update_branch", lambda *args: head)

    context = {
        "head_sha": head,
        "head_ref": "triage/issue-1",
        "review_id": 7,
        "timeline": [
            {
                "id": 7,
                "author": workflow.NAILAOPUS,
                "threads": [{"id": "thread", "comments": [{"id": 13}]}],
            }
        ],
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context))
    (tmp_path / "publish.json").write_text(
        json.dumps({"responses": {"thread": {"status": "resolved", "body": "Fixed"}}})
    )
    (tmp_path / "fix.patch").touch()
    # The reply and closure succeeded before the review request was attempted.
    reply = {
        "user": {"id": workflow.APP_ID},
        "body": workflow.marker(workflow.NAILAOPUS, "7", "thread"),
    }
    request = f"/review\n\n{workflow.marker(workflow.NAILAOPUS, '7', 'review-request')}"
    comments: list[dict[str, Any]] = [{"user": {"id": 0}, "body": request}]
    endpoints = {
        "repos/mlflow/mlflow/pulls/1/comments": [reply],
        "repos/mlflow/mlflow/issues/1/comments": comments,
    }
    monkeypatch.setattr(workflow, "list_items", lambda endpoint: endpoints[endpoint])
    monkeypatch.setattr(workflow, "threads", lambda *args: [{"id": "thread", "isResolved": True}])
    pr = {
        "state": "open",
        "draft": False,
        "user": {"id": workflow.APP_ID, "login": workflow.APP, "type": "Bot"},
        "head": {"sha": head, "ref": context["head_ref"], "repo": {"full_name": "mlflow/mlflow"}},
        "base": {"ref": "master"},
    }

    def api(endpoint: str, *, data: dict[str, Any] | None = None) -> Any:
        if endpoint == "repos/mlflow/mlflow/pulls/1":
            return pr
        if endpoint == "repos/mlflow/mlflow/pulls/1/reviews/7":
            return {"state": "COMMENTED", "user": {"login": workflow.NAILAOPUS}}
        assert endpoint == "repos/mlflow/mlflow/issues/1/comments"
        assert data == {"body": request}
        comments.append({"user": {"id": workflow.APP_ID}, **data})
        # The review workflow appends a run link, moving the marker off the last line.
        comments[-1]["body"] += "\n\n---\n\nNaiLaOpus running..."
        raise RuntimeError("Response lost after GitHub accepted the comment")

    monkeypatch.setattr(workflow, "api", api)
    with pytest.raises(RuntimeError, match="Response lost"):
        workflow.publish("mlflow/mlflow", "1", "7", context_path, tmp_path)
    workflow.publish("mlflow/mlflow", "1", "7", context_path, tmp_path)
    assert len(comments) == 2
