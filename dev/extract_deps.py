import ast
import re
from pathlib import Path


def parse_dependencies(content: str) -> list[str]:
    pattern = r"dependencies\s*=\s*(\[[\s\S]*?\])"
    match = re.search(pattern, content)
    deps_str = match.group(1)
    return ast.literal_eval(deps_str)


def main():
    content = Path("pyproject.toml").read_text()
    dependencies = parse_dependencies(content)
    print("\n".join(dependencies))


if __name__ == "__main__":
    main()
