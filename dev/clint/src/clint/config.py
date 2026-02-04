import re
import typing
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
    if non_existing_paths := [path for path in exclude_paths if not (repo_root / path).exists()]:
        raise ValueError(
            f"Non-existing paths found in exclude field: {non_existing_paths}. "
            f"All paths in the exclude list must exist."
        )


def _validate_typing_extensions_allowlist(allowlist: list[str]) -> None:
    """Validate that the typing-extensions-allowlist doesn't contain stdlib items.

    Args:
        allowlist: List of typing_extensions items to validate

    Raises:
        ValueError: If any item in the allowlist is available in stdlib typing
    """
    if not allowlist:
        return

    # Extract the item name from full paths like "typing_extensions.overload"
    # and check if it's available in stdlib typing
    stdlib_items = []
    for item in allowlist:
        name = item.rsplit(".", 1)[-1]
        if hasattr(typing, name):
            stdlib_items.append(item)

    if stdlib_items:
        raise ValueError(
            f"Items in typing-extensions-allowlist are available in stdlib `typing`: "
            f"{stdlib_items}. Use `from typing import ...` instead."
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

        typing_extensions_allowlist = clint.get("typing-extensions-allowlist", [])
        _validate_typing_extensions_allowlist(typing_extensions_allowlist)

        return cls(
            select=select,
            exclude=exclude_paths,
            forbidden_top_level_imports=clint.get("forbidden-top-level-imports", {}),
            typing_extensions_allowlist=typing_extensions_allowlist,
            example_rules=clint.get("example-rules", []),
            per_file_ignores=per_file_ignores,
        )
