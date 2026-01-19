import ast
import re
from pathlib import Path
from typing import cast


def parse_dependencies(content: str) -> list[str]:
    pattern = r"dependencies\s*=\s*(\[[\s\S]*?\])"
    match = re.search(pattern, content)
    if match is None:
        raise ValueError("Could not find dependencies in pyproject.toml")
    deps_str = match.group(1)
    return cast(list[str], ast.literal_eval(deps_str))


def main() -> None:
    content = Path("pyproject.toml").read_text()
    dependencies = parse_dependencies(content)
    print("\n".join(dependencies))


if __name__ == "__main__":
    main()
