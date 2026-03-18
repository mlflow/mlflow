from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

# Max length for skill names per the Agent Skills specification.
_MAX_SKILL_NAME_LENGTH = 64


@dataclass
class Skill:
    """A parsed skill loaded from a SKILL.md file.

    A skill is a self-contained unit of domain knowledge that judges can
    reference during evaluation. Each skill has a name, description, body
    content (from SKILL.md), and optional companion files.

    Attributes:
        name: Unique identifier for the skill (lowercase alphanumeric with hyphens).
        description: Short summary of what the skill covers.
        path: Absolute path to the skill directory on disk, or ``None`` for
            skills reconstructed from serialized data.
        metadata: Arbitrary key-value pairs from the SKILL.md frontmatter.
        body: The markdown body content of SKILL.md (after frontmatter).
        files: Mapping of relative file paths to their text content for all
            companion files in the skill directory (excludes SKILL.md itself).
    """

    name: str
    description: str
    path: Path | None
    body: str
    metadata: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


class SkillSet:
    """An ordered collection of skills that can be attached to a judge.

    SkillSet loads and indexes skills from filesystem paths, making them
    available for lookup by name during judge evaluation.

    Args:
        paths: List of paths to skill directories (containing SKILL.md)
            or direct paths to SKILL.md files.
    """

    def __init__(self, paths: list[str | Path]):
        self.skills: list[Skill] = [load_skill(p) for p in paths]

    def get_skill(self, name: str) -> Skill | None:
        """Look up a skill by name.

        Returns:
            The matching Skill, or None if no skill has the given name.
        """
        return next((s for s in self.skills if s.name == name), None)

    @classmethod
    def from_contents(cls, contents: list[dict[str, Any]]) -> SkillSet:
        """Reconstruct a SkillSet from serialized skill content dicts.

        Each dict should have keys: name, description, metadata, body, files.
        This is the inverse of the serialization done in
        ``InstructionsJudge.model_dump()``.
        """
        instance = cls.__new__(cls)
        skills = []
        for c in contents:
            name = c["name"]
            description = c["description"]
            metadata = c.get("metadata", {})
            body = c.get("body", "")
            files = c.get("files", {})

            _validate_name(name)

            skills.append(
                Skill(
                    name=name,
                    description=description,
                    path=None,
                    metadata=metadata,
                    body=body,
                    files=files,
                )
            )
        instance.skills = skills
        return instance

    def to_prompt(self) -> str:
        """Render the skill set as a prompt fragment for the judge's system message."""
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

    if not body.strip():
        raise MlflowException(
            "SKILL.md must have non-empty body content after the frontmatter.",
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
    """Validate a skill name against the Agent Skills specification.

    Follows the reference implementation at
    https://github.com/agentskills/agentskills/tree/main/skills-ref
    which uses NFKC normalization and ``str.isalnum()`` for Unicode support.
    """
    if not name or not isinstance(name, str) or not name.strip():
        raise MlflowException(
            "SKILL.md frontmatter must include a non-empty 'name' field.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    name = unicodedata.normalize("NFKC", name.strip())

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        raise MlflowException(
            f"Skill name must be at most {_MAX_SKILL_NAME_LENGTH} characters, got {len(name)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if name != name.lower():
        raise MlflowException(
            f"Skill name must be lowercase: '{name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if name.startswith("-") or name.endswith("-"):
        raise MlflowException(
            f"Skill name must not start or end with a hyphen: '{name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "--" in name:
        raise MlflowException(
            f"Skill name must not contain consecutive hyphens: '{name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not all(c.isalnum() or c == "-" for c in name):
        raise MlflowException(
            f"Skill name contains invalid characters: '{name}'. "
            f"Only letters, digits, and hyphens are allowed.",
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
