from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent, ScalarEvent

from mlflow.genai.skill_content.errors import content_unreadable, invalid_content
from mlflow.utils.validation import _validate_skill_name

SKILL_MANIFEST_FILE = "SKILL.md"
_MERGE_TAG = "tag:yaml.org,2002:merge"

_FRONTMATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)(.*)\Z", re.DOTALL)


def _reject_yaml_references(text: str) -> None:
    """
    Refuse frontmatter that uses YAML aliases or merge keys.

    ``safe_load`` still expands aliases, and nested merges double the intermediate mapping at
    every level, so a few hundred bytes of frontmatter can take minutes to load. Walking the
    parser events constructs nothing, and skill frontmatter never needs references. A merge
    key is rejected whether it refers to an alias or to an inline mapping, since either lets
    a manifest field come from somewhere other than where it appears to be declared.
    """
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        if isinstance(event, AliasEvent):
            raise yaml.YAMLError("YAML aliases and merge keys are not allowed in frontmatter")
        # A merge is either a plain, untagged ``<<`` or any scalar explicitly tagged ``!!merge``;
        # a quoted ``"<<"`` is an ordinary key.
        if isinstance(event, ScalarEvent) and (
            event.tag == _MERGE_TAG
            or (event.value == "<<" and event.style is None and event.tag is None)
        ):
            raise yaml.YAMLError("YAML aliases and merge keys are not allowed in frontmatter")


@dataclass(frozen=True)
class SkillManifest:
    """Content-derived metadata read from a skill directory's ``SKILL.md``."""

    name: str
    description: str | None
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
        _reject_yaml_references(match.group(1))
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise invalid_content(f"{SKILL_MANIFEST_FILE} frontmatter is not valid YAML: {e}")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} frontmatter must be a YAML mapping.")
    return metadata, match.group(2)


def inspect_skill_dir(
    root: str | os.PathLike[str], *, fallback_name: str | None = None
) -> SkillManifest:
    """
    Read the content-derived fields of the skill rooted at ``root``.

    The directory must contain a ``SKILL.md`` regular file. The skill name is declared in the
    frontmatter ``name`` field; the directory name is never used because fetched content lands
    in an arbitrary temporary directory. Import adapters that synthesize names for legacy
    layouts may pass ``fallback_name`` explicitly. The name is validated against the Agent
    Skills naming rules, and ``description`` is read when present. Any other frontmatter
    key is ignored, whatever its shape, so a ``SKILL.md`` written for other tooling still
    inspects cleanly.
    """
    root_path = Path(root)
    manifest_path = root_path / SKILL_MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise invalid_content(f"'{root_path}' does not contain a {SKILL_MANIFEST_FILE} file.")
    try:
        content = manifest_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise invalid_content(f"{SKILL_MANIFEST_FILE} in '{root_path}' is not valid UTF-8: {e}")
    except OSError as e:
        raise content_unreadable(manifest_path, e)
    metadata, _ = parse_skill_md(content)

    name = metadata.get("name")
    if name is None:
        name = fallback_name
    if name is None:
        raise invalid_content(
            f"{SKILL_MANIFEST_FILE} in '{root_path}' must declare a 'name' in its frontmatter."
        )
    if not isinstance(name, str):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} name must be a string, got {name!r}.")
    _validate_skill_name(name)

    description = metadata.get("description")
    if description is not None and not isinstance(description, str):
        raise invalid_content(f"{SKILL_MANIFEST_FILE} description must be a string.")

    return SkillManifest(
        name=name,
        description=description,
        path=root_path,
    )
