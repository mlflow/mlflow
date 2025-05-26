from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomli


@dataclass
class Config:
    exclude: list[str] = field(default_factory=list)
    # Path -> List of modules that should not be imported globally under that path
    forbidden_top_level_imports: dict[str, list[str]] = field(default_factory=dict)
    typing_extensions_allowlist: list[str] = field(default_factory=list)
    example_rules: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> Config:
        pyproject = Path("pyproject.toml")
        if not pyproject.exists():
            return cls()

        with pyproject.open("rb") as f:
            data = tomli.load(f)

        clint = data.get("tool", {}).get("clint", {})
        if not clint:
            return cls()

        return cls(
            exclude=clint.get("exclude", []),
            forbidden_top_level_imports=clint.get("forbidden-top-level-imports", {}),
            typing_extensions_allowlist=clint.get("typing-extensions-allowlist", []),
            example_rules=clint.get("example-rules", []),
        )
