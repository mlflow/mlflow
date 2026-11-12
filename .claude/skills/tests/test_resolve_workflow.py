import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def workflow() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_workflow", ROOT / ".claude/skills/resolve/scripts/workflow.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(workflow: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PR_CHECKOUT", str(repo))
    workflow.git("init", "--quiet")
    for name in ("mlflow/existing.py", "outside.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    workflow.git("add", ".")
    workflow.git(
        "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"
    )
    return repo


def patch_from_index(workflow: ModuleType, tmp_path: Path) -> Path:
    patch = tmp_path / "fix.patch"
    patch.write_text(workflow.git("diff", "--cached", "--binary") + "\n")
    workflow.git("reset", "--hard", "HEAD")
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
    workflow: ModuleType, checkout: Path, tmp_path: Path, paths: tuple[str, ...], allowed: bool
) -> None:
    for name in paths:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("updated\n")
    workflow.git("add", "--", *paths)
    patch = patch_from_index(workflow, tmp_path)
    if allowed:
        workflow.apply_patch(patch)
        assert set(workflow.git("diff", "--cached", "--name-only").splitlines()) == set(paths)
    else:
        with pytest.raises(ValueError, match="outside patch-paths.txt"):
            workflow.apply_patch(patch)


@pytest.mark.parametrize(
    ("source", "destination"),
    [("mlflow/existing.py", "renamed.py"), ("outside.py", "mlflow/renamed.py")],
)
def test_apply_patch_checks_both_sides_of_renames(
    workflow: ModuleType, checkout: Path, tmp_path: Path, source: str, destination: str
) -> None:
    (checkout / source).rename(checkout / destination)
    workflow.git("add", "--all", "--", source, destination)
    patch = patch_from_index(workflow, tmp_path)
    assert "rename from" in patch.read_text()
    with pytest.raises(ValueError, match="outside patch-paths.txt"):
        workflow.apply_patch(patch)


def test_patch_scope_parsing_matches_triage(
    workflow: ModuleType, checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = checkout / ".claude/skills/patch-paths.txt"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(" \t# comment\r\n\r\n \t \r\n\tmlflow/ \t\r\n\u2003docs/\u2003\r\n".encode())
    monkeypatch.setattr(workflow, "PATCH_PATHS", policy)
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
    workflow.git("add", "--", "mlflow/existing.py", "docs/x.md")
    workflow.apply_patch(patch_from_index(workflow, tmp_path))
