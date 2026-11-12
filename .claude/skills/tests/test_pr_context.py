import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def pr_context() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pr_context", ROOT / ".claude/scripts/pr_context.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(id: int, submitted_at: str, body: str = "", state: str = "COMMENTED") -> dict[str, Any]:
    return {
        "id": id,
        "user": {"login": "alice"},
        "state": state,
        "submitted_at": submitted_at,
        "body": body,
        "html_url": f"https://example.com/reviews/{id}",
    }


def review_comment(
    id: int, review_id: int, created_at: str, in_reply_to_id: int | None = None
) -> dict[str, Any]:
    comment = {
        "id": id,
        "node_id": f"node-{id}",
        "user": {"login": "alice"},
        "created_at": created_at,
        "body": f"comment {id}",
        "pull_request_review_id": review_id,
        "diff_hunk": "@@ -1 +1 @@",
    }
    if in_reply_to_id is not None:
        comment["in_reply_to_id"] = in_reply_to_id
    return comment


def thread(root_id: int, *, resolved: bool = False) -> dict[str, Any]:
    return {
        "id": f"thread-{root_id}",
        "path": "foo.py",
        "line": None,
        "originalLine": 7,
        "isResolved": resolved,
        "isOutdated": True,
        "comments": {"nodes": [{"id": f"node-{root_id}"}]},
    }


def test_describe_pr(pr_context: ModuleType) -> None:
    pr = {
        "number": 1,
        "html_url": "https://example.com/pull/1",
        "title": "Fix foo",
        "body": None,
        "state": "open",
        "draft": False,
        "user": {"login": "alice", "type": "User"},
        "author_association": "MEMBER",
        "base": {"ref": "master", "sha": "a" * 40},
        "head": {"ref": "fix-foo", "sha": "b" * 40, "repo": None},
    }
    assert pr_context.describe_pr(pr) == {
        "number": 1,
        "url": "https://example.com/pull/1",
        "title": "Fix foo",
        "body": "",
        "state": "open",
        "draft": False,
        "author": {"login": "alice", "type": "User", "association": "MEMBER"},
        "base": {"ref": "master", "sha": "a" * 40},
        "head": {"ref": "fix-foo", "sha": "b" * 40, "repo": ""},
    }


def test_threads_appear_under_the_review_that_started_them(pr_context: ModuleType) -> None:
    timeline = pr_context.build_timeline(
        issue_comments=[{"user": None, "created_at": "2026-01-02T00:00:00Z", "body": "hi"}],
        reviews=[
            review(100, "2026-01-01T00:00:00Z", body="summary"),
            # An empty review created by a reply; GitHub does not show it.
            review(200, "2026-01-03T00:00:00Z"),
        ],
        review_comments=[
            review_comment(1, 100, "2026-01-01T00:00:00Z"),
            review_comment(2, 200, "2026-01-03T00:00:00Z", in_reply_to_id=1),
        ],
        threads=[thread(1, resolved=True)],
    )
    assert timeline == [
        {
            "type": "review",
            "id": 100,
            "author": "alice",
            "state": "COMMENTED",
            "created_at": "2026-01-01T00:00:00Z",
            "body": "summary",
            "url": "https://example.com/reviews/100",
            "threads": [
                {
                    "id": "thread-1",
                    "path": "foo.py",
                    "line": 7,
                    "resolved": True,
                    "outdated": True,
                    "diff_hunk": "@@ -1 +1 @@",
                    "comments": [
                        {
                            "id": 1,
                            "author": "alice",
                            "created_at": "2026-01-01T00:00:00Z",
                            "body": "comment 1",
                            "review_id": 100,
                        },
                        {
                            "id": 2,
                            "author": "alice",
                            "created_at": "2026-01-03T00:00:00Z",
                            "body": "comment 2",
                            "review_id": 200,
                        },
                    ],
                }
            ],
        },
        {"type": "comment", "author": "", "created_at": "2026-01-02T00:00:00Z", "body": "hi"},
    ]


@pytest.mark.parametrize(
    ("reviews", "review_id", "expected"),
    [
        ([review(1, "2026-01-01T00:00:00Z")], None, []),
        ([review(1, "2026-01-01T00:00:00Z")], 1, [1]),
        ([review(1, "2026-01-01T00:00:00Z", state="APPROVED")], None, [1]),
        ([review(1, "2026-01-01T00:00:00Z", body="x")], None, [1]),
    ],
)
def test_empty_comment_reviews_are_hidden_unless_requested(
    pr_context: ModuleType,
    reviews: list[dict[str, Any]],
    review_id: int | None,
    expected: list[int],
) -> None:
    timeline = pr_context.build_timeline(
        issue_comments=[], reviews=reviews, review_comments=[], threads=[], review_id=review_id
    )
    assert [item["id"] for item in timeline] == expected
