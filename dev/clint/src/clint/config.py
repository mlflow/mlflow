import re
from dataclasses import dataclass, field

import tomli
from typing_extensions import Self

from clint.rules import ALL_RULES
from clint.utils import get_repo_root


def _validate_exclude_paths(exclude_paths: list[str]) -> None:
    """Validate that all paths in the exclude list exist.

    Args:
        exclude_paths: List of file/directory paths to validate (relative to repo root)

    Raises:
        ValueError: If any path in the exclude list does not exist
    """
    if not exclude_paths:
        return

    repo_root = get_repo_root()
    non_existing_paths = [path for path in exclude_paths if not (repo_root / path).exists()]

    if non_existing_paths:
        raise ValueError(
            f"Non-existing paths found in exclude field: {non_existing_paths}. "
            f"All paths in the exclude list must exist."
        )


@dataclass
class Config:
    select: set[str] = field(default_factory=set)
    exclude: list[str] = field(default_factory=list)
    # Path -> List of modules that should not be imported globally under that path
    forbidden_top_level_imports: dict[str, list[str]] = field(default_factory=dict)
    typing_extensions_allowlist: list[str] = field(default_factory=list)
    example_rules: list[str] = field(default_factory=list)
    # Compiled regex pattern -> Set of rule names to ignore for files matching the pattern
    per_file_ignores: dict[re.Pattern[str], set[str]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Self:
        repo_root = get_repo_root()
        pyproject = repo_root / "pyproject.toml"
        if not pyproject.exists():
            return cls()

        with pyproject.open("rb") as f:
            data = tomli.load(f)

        clint = data.get("tool", {}).get("clint", {})
        if not clint:
            return cls()

        per_file_ignores_raw = clint.get("per-file-ignores", {})
        per_file_ignores: dict[re.Pattern[str], set[str]] = {}
        for pattern, rules in per_file_ignores_raw.items():
            per_file_ignores[re.compile(pattern)] = set(rules)

        select = clint.get("select")
        if select is None:
            select = ALL_RULES
        else:
            if unknown_rules := set(select) - ALL_RULES:
                raise ValueError(f"Unknown rules in 'select': {unknown_rules}")
            select = set(select)

        exclude_paths = clint.get("exclude", [])
        _validate_exclude_paths(exclude_paths)

        return cls(
            select=select,
            exclude=exclude_paths,
            forbidden_top_level_imports=clint.get("forbidden-top-level-imports", {}),
            typing_extensions_allowlist=clint.get("typing-extensions-allowlist", []),
            example_rules=clint.get("example-rules", []),
            per_file_ignores=per_file_ignores,
        )
