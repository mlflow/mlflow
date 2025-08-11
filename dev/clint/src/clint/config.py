import re
from dataclasses import dataclass, field
from pathlib import Path

import tomli
from typing_extensions import Self

from clint.rules import ALL_RULES


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
        pyproject = Path("pyproject.toml")
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

        return cls(
            select=select,
            exclude=clint.get("exclude", []),
            forbidden_top_level_imports=clint.get("forbidden-top-level-imports", {}),
            typing_extensions_allowlist=clint.get("typing-extensions-allowlist", []),
            example_rules=clint.get("example-rules", []),
            per_file_ignores=per_file_ignores,
        )
