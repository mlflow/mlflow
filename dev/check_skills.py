import re
import sys
from pathlib import Path
from typing import Any

import yaml

# https://agentskills.io/specification#frontmatter
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NAME_MAX = 64
DESCRIPTION_MAX = 1024


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    data = yaml.safe_load(text[4:end]) or {}
    return data if isinstance(data, dict) else None


def validate_name(name: Any, parent: str) -> list[str]:
    if not name:
        return ["missing or empty `name`"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or len(name) > NAME_MAX:
        return [
            f"invalid `name`: must be 1-{NAME_MAX} lowercase alphanumeric/hyphen chars, "
            "no leading/trailing or consecutive hyphens"
        ]
    if name != parent:
        return [f"`name` {name!r} does not match parent directory {parent!r}"]
    return []


def validate_description(description: Any) -> list[str]:
    if not description:
        return ["missing or empty `description`"]
    if not isinstance(description, str) or len(description) > DESCRIPTION_MAX:
        return [f"invalid `description`: must be 1-{DESCRIPTION_MAX} characters"]
    return []


def check(path: Path) -> list[str]:
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    if fm is None:
        return ["missing frontmatter"]
    return [
        *validate_name(fm.get("name"), path.parent.name),
        *validate_description(fm.get("description")),
    ]


def main(argv: list[str]) -> bool:
    failed = False
    for arg in argv:
        for err in check(Path(arg)):
            print(f"{arg}: {err}", file=sys.stderr)
            failed = True
    return failed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
