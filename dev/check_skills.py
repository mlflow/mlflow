import sys
from pathlib import Path

import yaml

REQUIRED = ("name", "description")


def parse_frontmatter(text: str) -> dict[str, object] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    data = yaml.safe_load(text[4:end]) or {}
    return data if isinstance(data, dict) else None


def check(path: Path) -> list[str]:
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    if fm is None:
        return ["missing YAML frontmatter"]
    return [f"missing or empty `{k}`" for k in REQUIRED if not fm.get(k)]


def main(argv: list[str]) -> int:
    any_failed = False
    for arg in argv:
        for err in check(Path(arg)):
            print(f"{arg}: {err}", file=sys.stderr)
            any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
