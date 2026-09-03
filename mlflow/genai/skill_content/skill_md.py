from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlflow.genai.skill_content.errors import invalid_content
from mlflow.utils.validation import _validate_skill_name

SKILL_MANIFEST_FILE = "SKILL.md"

_FRONTMATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class SkillManifest:
    """Content-derived metadata read from a skill directory's ``SKILL.md``."""

    name: str
    description: str | None
    keywords: tuple[str, ...]
    path: Path


def parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
    """
    Split ``SKILL.md`` content into its YAML frontmatter and markdown body.

    Content without a leading ``---`` block has empty frontmatter. A frontmatter block that is
    opened but not closed, is not valid YAML, or is not a mapping is rejected rather than
    silently ignored, because the registry derives identity from it.
    """
    if not content.startswith("---"):
        return {}, content
    match = _FRONTMATTER_PATTERN.match(content)
    if match is None:
        raise invalid_content(f"{SKILL_MANIFEST_FILE} frontmatter is not closed with '---'.")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise invalid_content(f"{SKILL_MANIFEST_FILE} frontmatter is not valid YAML: {e}")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} frontmatter must be a YAML mapping.")
    return metadata, match.group(2)


def _extract_keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("keywords")
    if raw is None and isinstance(metadata.get("metadata"), dict):
        raw = metadata["metadata"].get("keywords")
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        raise invalid_content(
            f"{SKILL_MANIFEST_FILE} keywords must be a list or comma-separated string."
        )
    keywords = []
    for item in raw:
        if not isinstance(item, (str, int, float)):
            raise invalid_content(f"{SKILL_MANIFEST_FILE} keywords must be strings, got {item!r}.")
        if value := str(item).strip():
            keywords.append(value)
    return tuple(keywords)


def inspect_skill_dir(root: str | os.PathLike[str]) -> SkillManifest:
    """
    Read the content-derived fields of the skill rooted at ``root``.

    The directory must contain a ``SKILL.md`` regular file. The skill name comes from the
    frontmatter ``name`` field and falls back to the directory name, then is validated against
    the Agent Skills naming rules. ``description`` and ``keywords`` are read from the
    frontmatter when present.
    """
    root_path = Path(root)
    manifest_path = root_path / SKILL_MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise invalid_content(f"'{root_path}' does not contain a {SKILL_MANIFEST_FILE} file.")
    try:
        content = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise invalid_content(f"{SKILL_MANIFEST_FILE} in '{root_path}' is not valid UTF-8: {e}")
    metadata, _ = parse_skill_md(content)

    name = metadata.get("name")
    if name is None:
        name = root_path.name
    if not isinstance(name, str):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} name must be a string, got {name!r}.")
    _validate_skill_name(name)

    description = metadata.get("description")
    if description is not None and not isinstance(description, str):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} description must be a string.")

    return SkillManifest(
        name=name,
        description=description,
        keywords=_extract_keywords(metadata),
        path=root_path,
    )
