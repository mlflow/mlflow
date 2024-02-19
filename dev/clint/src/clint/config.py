from __future__ import annotations

from dataclasses import dataclass

import tomli


@dataclass
class Config:
    exclude: list[str]

    @staticmethod
    def load() -> Config:
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
            exclude = data["tool"]["clint"]["exclude"]
            return Config(exclude)
