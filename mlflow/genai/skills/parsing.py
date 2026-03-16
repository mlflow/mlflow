from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    metadata: dict[str, str] = field(default_factory=dict)
    body: str = ""
    files: dict[str, str] = field(default_factory=dict)


class SkillSet:
    def __init__(self, paths: list[str | Path]):
        self.skills: list[Skill] = [load_skill(p) for p in paths]

    def get_skill(self, name: str) -> Skill | None:
        return next((s for s in self.skills if s.name == name), None)

    def to_prompt(self) -> str:
        from mlflow.genai.skills.prompt_format import to_prompt

        return to_prompt(self.skills)


def load_skill(path: str | Path) -> Skill:
    path = Path(path)
    if path.is_file() and path.name == "SKILL.md":
        skill_dir = path.parent
        skill_file = path
    elif path.is_dir():
        skill_dir = path
        skill_file = path / "SKILL.md"
    else:
        raise MlflowException(
            f"Invalid skill path: {path}. Must be a directory containing SKILL.md "
            f"or a direct path to a SKILL.md file.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if not skill_file.exists():
        raise MlflowException(
            f"SKILL.md not found at {skill_file}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    content = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content)

    name = frontmatter.get("name")
    _validate_name(name)

    description = frontmatter.get("description")
    if not description:
        raise MlflowException(
            "SKILL.md frontmatter must include a non-empty 'description' field.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    metadata = frontmatter.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    files = _load_files(skill_dir)

    return Skill(
        name=name,
        description=description,
        path=skill_dir.resolve(),
        metadata={str(k): str(v) for k, v in metadata.items()},
        body=body.strip(),
        files=files,
    )


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---"):
        raise MlflowException(
            "SKILL.md must start with YAML frontmatter (---).",
            error_code=INVALID_PARAMETER_VALUE,
        )
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise MlflowException(
            "SKILL.md frontmatter must be enclosed by --- delimiters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    return frontmatter, body


def _validate_name(name: str | None) -> None:
    if not name:
        raise MlflowException(
            "SKILL.md frontmatter must include a non-empty 'name' field.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if len(name) > 64:
        raise MlflowException(
            f"Skill name must be at most 64 characters, got {len(name)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "--" in name:
        raise MlflowException(
            f"Skill name must not contain consecutive hyphens: '{name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not _NAME_PATTERN.match(name):
        raise MlflowException(
            f"Skill name must be lowercase alphanumeric with hyphens, "
            f"and must not start or end with a hyphen: '{name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _load_files(skill_dir: Path) -> dict[str, str]:
    result = {}
    for f in sorted(skill_dir.rglob("*")):
        if not f.is_file() or f.name == "SKILL.md":
            continue
        try:
            rel = str(f.relative_to(skill_dir))
            result[rel] = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
    return result
