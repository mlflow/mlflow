import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock, call

import jsonschema  # type: ignore[import-untyped]
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RESOLVE_CLI = ROOT / ".claude/skills/resolve/scripts/cli.py"


@pytest.fixture
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resolve_cli", RESOLVE_CLI)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("operation", ["collect", "validate", "embed", "publish"])
def test_operation_help(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_PATH", raising=False)
    monkeypatch.delenv("OUT_DIR", raising=False)
    result = subprocess.run(
        [sys.executable, RESOLVE_CLI, operation, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"usage: cli.py {operation}" in result.stdout


@pytest.fixture
def checkout(cli: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PR_CHECKOUT", str(repo))
    cli.git("init", "--quiet")
    for name in ("mlflow/existing.py", "outside.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    cli.git("add", ".")
    cli.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base")
    return repo


def patch_from_index(cli: ModuleType, tmp_path: Path) -> Path:
    patch = tmp_path / "fix.patch"
    patch.write_text(cli.git("diff", "--cached", "--binary") + "\n")
    cli.git("reset", "--hard", "HEAD")
    return patch


@pytest.mark.parametrize(
    ("paths", "allowed"),
    [
        (("mlflow/x.py", "tests/x.py", "docs/x.md", "libs/typescript/x.ts"), True),
        ((".github/actions/a.yml",), False),
        (("pyproject.toml",), False),
        (("uv.lock",), False),
        (("dev/tool.py",), False),
        (("mlflow-other/x.py",), False),
        (("mlflow/x.py", "pyproject.toml"), False),
    ],
)
def test_apply_patch_path_scope(
    cli: ModuleType, checkout: Path, tmp_path: Path, paths: tuple[str, ...], allowed: bool
) -> None:
    for name in paths:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("updated\n")
    cli.git("add", "--", *paths)
    patch = patch_from_index(cli, tmp_path)
    if allowed:
        cli.apply_patch(patch)
        assert set(cli.git("diff", "--cached", "--name-only").splitlines()) == set(paths)
    else:
        with pytest.raises(ValueError, match="outside patch-paths.txt"):
            cli.apply_patch(patch)


@pytest.mark.parametrize(
    ("source", "destination"),
    [("mlflow/existing.py", "renamed.py"), ("outside.py", "mlflow/renamed.py")],
)
def test_apply_patch_checks_both_sides_of_renames(
    cli: ModuleType, checkout: Path, tmp_path: Path, source: str, destination: str
) -> None:
    (checkout / source).rename(checkout / destination)
    cli.git("add", "--all", "--", source, destination)
    patch = patch_from_index(cli, tmp_path)
    assert "rename from" in patch.read_text()
    with pytest.raises(ValueError, match="outside patch-paths.txt"):
        cli.apply_patch(patch)


def test_patch_scope_parsing_matches_triage(
    cli: ModuleType, checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = checkout / ".claude/skills/patch-paths.txt"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(" \t# comment\r\n\r\n \t \r\n\tmlflow/ \t\r\n\u2003docs/\u2003\r\n".encode())
    monkeypatch.setattr(cli, "PATCH_PATHS", policy)
    config = yaml.safe_load((ROOT / ".github/workflows/triage.yml").read_text())
    step = next(s for s in config["jobs"]["issue"]["steps"] if s.get("id") == "policy")
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=checkout,
        env={
            "GITHUB_OUTPUT": str(output),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        },
        check=True,
    )
    assert output.read_text() == "patch_path_allowlist<<EOF\nmlflow/\ndocs/\nEOF\n"
    for name in ("mlflow/existing.py", "docs/x.md"):
        path = checkout / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("updated\n")
    cli.git("add", "--", "mlflow/existing.py", "docs/x.md")
    cli.apply_patch(patch_from_index(cli, tmp_path))


@pytest.fixture
def metadata_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DEFAULT_BRANCH", "master")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("DRY_RUN", "false")
    path = tmp_path / "context.json"
    path.write_text(
        json.dumps({
            "head_ref": "triage/issue-1",
            "head_sha": "original-sha",
            "pr": {"title": "Before", "body": "Original description"},
            "review_id": 12,
            "timeline": [
                {
                    "id": 12,
                    "author": "maintainer",
                    "body": "",
                    "url": "https://github.com/mlflow/mlflow/pull/1#pullrequestreview-12",
                    "threads": [
                        {
                            "id": "PRRT_1",
                            "resolved": False,
                            "comments": [{"id": 34, "review_id": 12}],
                        }
                    ],
                }
            ],
        })
    )
    return path


@pytest.fixture
def resolve_schema() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        yaml.safe_load((ROOT / ".claude/skills/resolve/payload.schema.yml").read_text())
    )


def test_collect_includes_current_pr_metadata(
    cli: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pr = {
        "title": "Current title",
        "body": "Current **Markdown** body",
        "head": {
            "sha": "head-sha",
            "ref": "triage/issue-1",
            "repo": {"full_name": "mlflow/mlflow"},
        },
        "state": "open",
        "draft": False,
        "user": {"id": cli.APP_ID, "login": cli.APP, "type": "Bot"},
        "base": {"ref": "master"},
    }
    review = {
        "state": "CHANGES_REQUESTED",
        "user": {"login": cli.COPILOT, "type": "Bot"},
        "body": "",
    }

    def github_api(endpoint: str, *args: str) -> Any:
        if endpoint == "repos/mlflow/mlflow/pulls/1":
            return pr
        if endpoint == "repos/mlflow/mlflow/pulls/1/reviews/12":
            return review
        assert endpoint == "--paginate"
        return [[]]

    api = Mock(side_effect=github_api)
    timeline = [
        {
            "id": 12,
            "author": cli.COPILOT,
            "body": "",
            "threads": [{"id": "PRRT_1", "comments": [{"review_id": 12}]}],
        }
    ]
    fetch_timeline = Mock(return_value=timeline)
    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(cli, "fetch_timeline", fetch_timeline)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github-output"))
    monkeypatch.setenv("DEFAULT_BRANCH", "master")
    context_path = tmp_path / "context.json"

    cli.collect("mlflow/mlflow", "1", "12", context_path)

    assert json.loads(context_path.read_text())["pr"] == {
        "title": "Current title",
        "body": "Current **Markdown** body",
    }
    assert api.call_count == 4
    fetch_timeline.assert_called_once_with("mlflow/mlflow", "1", "12")


@pytest.mark.parametrize(
    "pr_update",
    [
        {},
        {"title": ""},
        {"title": " \n "},
        {"title": "Line one\nLine two"},
        {"title": "a" * 257},
        {"unknown": "value"},
    ],
)
def test_schema_rejects_malformed_pr_updates(
    resolve_schema: jsonschema.Draft202012Validator, pr_update: dict[str, str]
) -> None:
    payload = {
        "responses": {"PRRT_1": {"status": "resolved", "body": "Updated the PR."}},
        "pr_update": pr_update,
    }

    assert list(resolve_schema.iter_errors(payload))


@pytest.mark.parametrize(
    (
        "proposed",
        "current",
        "with_patch",
        "after_patch",
        "error",
        "expected_update",
    ),
    [
        ({"title": "After"}, {}, False, None, None, {"title": "After"}),
        ({"body": "Updated"}, {}, False, None, None, {"body": "Updated"}),
        (
            {"title": "After", "body": "Updated"},
            {},
            True,
            None,
            None,
            {"title": "After", "body": "Updated"},
        ),
        (
            {"title": "After", "body": "Updated"},
            {"title": "After"},
            False,
            None,
            None,
            {"body": "Updated"},
        ),
        ({}, {}, True, None, None, {}),
        ({}, {}, False, None, None, {}),
        ({"title": "After"}, {"title": "After"}, False, None, None, {}),
        ({"title": "After"}, {"title": "Concurrent"}, False, None, "PR title changed", {}),
        ({"title": "After"}, {}, True, "title", "PR title changed", {}),
        ({"title": "After"}, {}, True, "stale", None, {"title": "After"}),
        (
            {"title": "After"},
            {},
            True,
            "reverted",
            "PR head changed before the review request",
            {"title": "After"},
        ),
        ({"title": "After"}, {}, True, "head", "PR changed before publishing", {}),
        ({"title": "After"}, {}, True, "branch", "PR changed before publishing", {}),
        ({"body": "Updated"}, {}, False, None, "GitHub did not apply", {"body": "Updated"}),
    ],
    ids=[
        "title",
        "body",
        "mixed",
        "partially-applied",
        "patch-only",
        "already-addressed",
        "retry",
        "concurrent-edit",
        "concurrent-edit-after-patch",
        "stale-head",
        "branch-reverted-before-review-request",
        "concurrent-head-change",
        "concurrent-branch-change",
        "failed-write",
    ],
)
def test_publish_pr_update(
    cli: ModuleType,
    metadata_context: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposed: dict[str, str],
    current: dict[str, str],
    with_patch: bool,
    after_patch: str | None,
    error: str | None,
    expected_update: dict[str, str],
) -> None:
    out_dir = metadata_context.parent
    result: dict[str, Any] = {
        "responses": {"PRRT_1": {"status": "resolved", "body": "Addressed the feedback."}}
    }
    if proposed:
        result["pr_update"] = proposed
    (out_dir / "publish.json").write_text(json.dumps(result))
    context = json.loads(metadata_context.read_text())
    repo = "mlflow/mlflow"
    endpoint = f"repos/{repo}/pulls/1"
    review_endpoint = f"{endpoint}/reviews/12"
    reply_endpoint = f"{endpoint}/comments/34/replies"
    ref_endpoint = f"repos/{repo}/git/ref/heads/{context['head_ref']}"
    issue_comments = f"repos/{repo}/issues/1/comments"
    pr = {
        **context["pr"],
        **current,
        "state": "open",
        "draft": False,
        "user": {"id": cli.APP_ID, "login": cli.APP, "type": "Bot"},
        "base": {"ref": "master"},
        "head": {
            "ref": context["head_ref"],
            "sha": context["head_sha"],
            "repo": {"full_name": repo},
        },
    }
    review = {"state": "APPROVED", "user": {"login": "maintainer"}}
    expect_feedback = error is None or after_patch == "reverted"

    def github_api(
        endpoint_arg: str,
        *args: str,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> Any:
        if endpoint_arg == endpoint:
            if method == "PATCH" and expect_feedback:
                assert data is not None
                pr.update(data)
            return pr
        if endpoint_arg == ref_endpoint:
            sha = context["head_sha"] if after_patch == "reverted" else "new-sha"
            return {"object": {"sha": sha}}
        if endpoint_arg == review_endpoint:
            return review
        if endpoint_arg == "graphql":
            return {"data": {"resolveReviewThread": {"thread": {"isResolved": True}}}}
        if endpoint_arg == "--paginate":
            return [[]]
        assert endpoint_arg in {reply_endpoint, issue_comments}
        return None

    def update_branch(
        repo: str, review_id: str, context: dict[str, Any], head: str, tree: str
    ) -> str:
        if after_patch not in {"stale", "reverted"}:
            pr["head"]["sha"] = "different-sha" if after_patch == "head" else "new-sha"
        if after_patch == "title":
            pr["title"] = "Concurrent"
        if after_patch == "branch":
            pr["head"]["ref"] = "triage/another-branch"
        return "new-sha"

    api = Mock(side_effect=github_api)
    branch = Mock(side_effect=update_branch)
    git = Mock(return_value="staged-tree")
    timeline = Mock(return_value=context["timeline"])
    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(cli, "update_branch", branch)
    monkeypatch.setattr(cli, "git", git)
    monkeypatch.setattr(cli, "fetch_timeline", timeline)
    if with_patch:
        (out_dir / "fix.patch").touch()
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    if error:
        with pytest.raises(ValueError, match=error):
            cli.publish(repo, "1", "12", metadata_context, out_dir)
    else:
        cli.publish(repo, "1", "12", metadata_context, out_dir)

    if with_patch:
        git.assert_called_once_with("write-tree")
        branch.assert_called_once_with(repo, "12", context, "original-sha", "staged-tree")
    else:
        git.assert_not_called()
        branch.assert_not_called()
    writes = [entry for entry in api.call_args_list if entry.kwargs.get("method") == "PATCH"]
    assert writes == (
        [call(endpoint, data=expected_update, method="PATCH")] if expected_update else []
    )
    replies = [entry for entry in api.call_args_list if entry.args[0] == reply_endpoint]
    closures = [entry for entry in api.call_args_list if entry.args[0] == "graphql"]
    ref_reads = [entry for entry in api.call_args_list if entry.args[0] == ref_endpoint]
    assert ref_reads == ([call(ref_endpoint)] if after_patch in {"stale", "reverted"} else [])
    review_requests = [entry for entry in api.call_args_list if entry.args[0] == issue_comments]
    if error is None and with_patch:
        assert len(review_requests) == 1
        assert review_requests[0].kwargs["data"]["body"].startswith("/review\n")
    else:
        assert review_requests == []
    if expect_feedback:
        timeline.assert_called_once_with(repo, "1", "12")
        assert len(replies) == len(closures) == 1
        assert ("[Fix commit]" in str(replies[0].kwargs["data"]["body"])) == (
            with_patch and not proposed
        )
        if writes:
            assert api.call_args_list.index(writes[0]) < api.call_args_list.index(replies[0])
    else:
        timeline.assert_not_called()
        assert replies == closures == []


def test_publish_retries_with_embedded_description_after_update(
    cli: ModuleType, metadata_context: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = metadata_context.parent
    description = "![Screenshot](https://github.com/user-attachments/assets/id)"
    (out_dir / "publish.json").write_text(
        json.dumps({
            "responses": {"PRRT_1": {"status": "resolved", "body": "Addressed."}},
            "pr_update": {"body": description},
        })
    )
    context = json.loads(metadata_context.read_text())
    repo = "mlflow/mlflow"
    endpoint = f"repos/{repo}/pulls/1"
    reply_endpoint = f"{endpoint}/comments/34/replies"
    pr = {
        **context["pr"],
        "state": "open",
        "draft": False,
        "user": {"id": cli.APP_ID, "login": cli.APP, "type": "Bot"},
        "base": {"ref": "master"},
        "head": {
            "ref": context["head_ref"],
            "sha": context["head_sha"],
            "repo": {"full_name": repo},
        },
    }
    review = {"state": "APPROVED", "user": {"login": "maintainer"}}
    fail_reply = True

    def github_api(
        endpoint_arg: str,
        *args: str,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> Any:
        nonlocal fail_reply
        if endpoint_arg == endpoint:
            if method == "PATCH":
                assert data is not None
                pr.update(data)
            return pr
        if endpoint_arg == f"{endpoint}/reviews/12":
            return review
        if endpoint_arg == "--paginate":
            return [[]]
        if endpoint_arg == reply_endpoint:
            if fail_reply:
                fail_reply = False
                raise RuntimeError("temporary publication failure")
            return None
        if endpoint_arg == "graphql":
            return {"data": {"resolveReviewThread": {"thread": {"isResolved": True}}}}
        raise AssertionError(endpoint_arg)

    api = Mock(side_effect=github_api)
    timeline = Mock(return_value=context["timeline"])
    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(cli, "fetch_timeline", timeline)

    with pytest.raises(RuntimeError, match="temporary publication failure"):
        cli.publish(repo, "1", "12", metadata_context, out_dir)

    assert pr["body"] == description
    cli.publish(repo, "1", "12", metadata_context, out_dir)

    patches = [entry for entry in api.call_args_list if entry.kwargs.get("method") == "PATCH"]
    assert patches == [call(endpoint, data={"body": description}, method="PATCH")]
    assert len([entry for entry in api.call_args_list if entry.args[0] == reply_endpoint]) == 2
    timeline.assert_has_calls([call(repo, "1", "12"), call(repo, "1", "12")])


def test_preview_pr_update(
    cli: ModuleType, metadata_context: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = metadata_context.parent
    (out_dir / "payload.json").write_text(
        json.dumps({
            "responses": {"PRRT_1": {"status": "resolved", "body": "Updated description."}},
            "pr_update": {"body": "<new description>"},
        })
    )
    monkeypatch.setenv("DRY_RUN", "true")
    api = Mock()
    git = Mock(return_value="staged-tree")
    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(cli, "git", git)

    cli.publish("mlflow/mlflow", "1", "12", metadata_context, out_dir)

    api.assert_not_called()
    git.assert_not_called()
    result = (out_dir / "summary.md").read_text()
    assert "Original description" in result
    assert "&lt;new description&gt;" in result
    assert "Threads to close**\n\nPRRT_1" in result


@pytest.mark.parametrize("media_in_description", [False, True], ids=["reply", "description"])
def test_validate_and_embed_media(
    cli: ModuleType,
    metadata_context: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_in_description: bool,
) -> None:
    out_dir = metadata_context.parent
    media = out_dir / "media"
    media.mkdir()
    local_reference = f"![Screenshot]({media / 'screenshot.png'})"
    attachment = "![Screenshot](https://github.com/user-attachments/id)"
    reply = "Updated the description." if media_in_description else local_reference
    description = local_reference if media_in_description else "Description"
    payload = out_dir / "payload.json"
    payload.write_text(
        json.dumps({
            "responses": {"PRRT_1": {"status": "resolved", "body": reply}},
            "pr_update": {"title": "After", "body": description},
        })
    )
    command = Mock(side_effect=["", "", attachment])
    monkeypatch.setattr(cli, "command", command)
    monkeypatch.setenv("REPOSITORY_ID", "123")

    cli.validate(metadata_context, out_dir)
    cli.embed(out_dir)

    assert command.call_args_list == [
        call(
            "uvx",
            "check-jsonschema@0.37.4",
            "--schemafile",
            ".claude/skills/resolve/payload.schema.yml",
            payload,
        ),
        call(*cli.media_command(media), "--check", input=f"{reply}\n\n{description}"),
        call(
            *cli.media_command(media, strict=media_in_description),
            "--repository-id",
            "123",
            input=local_reference,
        ),
    ]
    assert json.loads((out_dir / "publish.json").read_text()) == {
        "responses": {
            "PRRT_1": {"status": "resolved", "body": reply.replace(local_reference, attachment)}
        },
        "pr_update": {"title": "After", "body": description.replace(local_reference, attachment)},
    }


def test_embed_stops_when_description_media_upload_fails(
    cli: ModuleType, metadata_context: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = metadata_context.parent
    media = out_dir / "media"
    media.mkdir()
    local_reference = f"![Screenshot]({media / 'screenshot.png'})"
    (out_dir / "payload.json").write_text(
        json.dumps({
            "responses": {"PRRT_1": {"status": "resolved", "body": "Updated."}},
            "pr_update": {"body": local_reference},
        })
    )
    command = Mock(
        side_effect=[
            "",
            "",
            subprocess.CalledProcessError(1, ["embed-media"]),
        ]
    )
    monkeypatch.setattr(cli, "command", command)
    monkeypatch.setenv("REPOSITORY_ID", "123")

    cli.validate(metadata_context, out_dir)
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero exit status 1"):
        cli.embed(out_dir)

    assert command.call_args_list[-1] == call(
        *cli.media_command(media, strict=True),
        "--repository-id",
        "123",
        input=local_reference,
    )
    assert not (out_dir / "publish.json").exists()


@pytest.mark.parametrize(
    ("status", "update", "with_patch", "error"),
    [
        ("resolved", {}, False, None),
        ("resolved", {"pr_update": {"title": "Before"}}, False, "must change each proposed field"),
        ("answered", {"pr_update": {"title": "After"}}, False, None),
        ("answered", {}, True, None),
        ("blocked", {}, True, None),
    ],
)
def test_validate_pr_update(
    cli: ModuleType,
    metadata_context: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    update: dict[str, dict[str, str]],
    with_patch: bool,
    error: str | None,
) -> None:
    out_dir = metadata_context.parent
    payload = out_dir / "payload.json"
    payload.write_text(
        json.dumps({"responses": {"PRRT_1": {"status": status, "body": "Reply."}}, **update})
    )
    command = Mock()
    monkeypatch.setattr(cli, "command", command)
    if with_patch:
        (out_dir / "fix.patch").touch()

    if error:
        with pytest.raises(ValueError, match=error):
            cli.validate(metadata_context, out_dir)
    else:
        cli.validate(metadata_context, out_dir)

    command.assert_any_call(
        "uvx",
        "check-jsonschema@0.37.4",
        "--schemafile",
        ".claude/skills/resolve/payload.schema.yml",
        payload,
    )
