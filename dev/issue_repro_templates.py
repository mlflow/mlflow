"""Safe, source-controlled reproduction commands for issue triage."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class InvalidTemplate(ValueError):
    """Raised when untrusted template data is outside the small allowed contract."""


@dataclass(frozen=True)
class ReproductionTemplate:
    id: str
    test_path: str | None = None


NO_REPRO = "no_repro"
TARGETED_PYTEST = "targeted_pytest"


def validate_template(value: object, repo_root: Path | None = None) -> ReproductionTemplate:
    """Parse a model selection without allowing it to influence a shell command."""
    if not isinstance(value, dict) or set(value) - {"id", "test_path"}:
        raise InvalidTemplate("template must contain only id and test_path")
    template_id = value.get("id")
    test_path = value.get("test_path")
    if template_id == NO_REPRO and test_path is None:
        return ReproductionTemplate(NO_REPRO)
    if template_id != TARGETED_PYTEST or not isinstance(test_path, str):
        raise InvalidTemplate("unknown reproduction template")
    _validate_test_path(test_path, repo_root)
    return ReproductionTemplate(TARGETED_PYTEST, test_path)


def _validate_test_path(test_path: str, repo_root: Path | None) -> None:
    path = Path(test_path)
    if (
        not test_path.startswith("tests/")
        or not test_path.endswith(".py")
        or path.is_absolute()
        or "\\" in test_path
        or ".." in path.parts
        or any(character.isspace() for character in test_path)
        or any(character in test_path for character in "'\"`$;&|<>*?()[]{}!")
    ):
        raise InvalidTemplate("test path is not a plain tests/*.py path")
    if repo_root is not None:
        candidate = repo_root / path
        if not candidate.is_file():
            raise InvalidTemplate("test path does not exist")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", test_path],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode:
            raise InvalidTemplate("test path is not tracked")


def command_for(template: ReproductionTemplate) -> list[str] | None:
    """Return the only command shape supported by the MVP, as an argv array."""
    if template.id == NO_REPRO:
        return None
    if template.id == TARGETED_PYTEST and template.test_path:
        return ["uv", "run", "pytest", template.test_path]
    raise InvalidTemplate("invalid template")
