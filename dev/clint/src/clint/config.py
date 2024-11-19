from __future__ import annotations

from dataclasses import dataclass

import tomli


@dataclass
class Config:
    exclude: list[str]
    # Path -> List of modules that should not be imported globally under that path
    forbidden_top_level_imports: dict[str, list[str]]

    @classmethod
    def load(cls) -> Config:
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
            exclude = data["tool"]["clint"]["exclude"]
            forbidden_imports = data["tool"]["clint"]["forbidden-top-level-imports"]
            return cls(exclude, forbidden_imports)
