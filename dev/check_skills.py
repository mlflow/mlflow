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
    data = yaml.safe_load(text[4:end])
    return data if isinstance(data, dict) else None


def check(path: Path) -> list[str]:
    if (fm := parse_frontmatter(path.read_text(encoding="utf-8"))) is None:
        return ["missing YAML frontmatter"]
    return [f"missing or empty `{k}`" for k in REQUIRED if not fm.get(k)]


def main(argv: list[str]) -> int:
    failed = 0
    for arg in argv:
        if errors := check(Path(arg)):
            failed += 1
            for err in errors:
                print(f"{arg}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
