from dataclasses import dataclass
from typing import Callable

import mlflow.proxy


@dataclass
class _Route:
    route: str
    function: Callable
    name: str = None
    description: str = None
    run_id: str = None
    cache_config = None
    limits_per_minute: int = 30
    log_destination: str = None
    active: bool = False

    def to_dict(self):
        return {
            name: getattr(self, name)
            for name in self.__annotations__
            if getattr(self, name) is not None
        }

    @classmethod
    def from_dict(cls, data):
        data["function"] = getattr(mlflow.proxy, data["function"])  # placeholder
        return cls(**data)
