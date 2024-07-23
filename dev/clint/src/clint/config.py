from __future__ import annotations

from dataclasses import dataclass

import tomli


@dataclass
class Config:
    exclude: list[str]

    @classmethod
    def load(cls) -> Config:
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
            exclude = data["tool"]["clint"]["exclude"]
            return cls(exclude)
