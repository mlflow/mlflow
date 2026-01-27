from __future__ import annotations

from typing import TYPE_CHECKING

from mlflow.demo.base import DemoFeature

if TYPE_CHECKING:
    from mlflow.demo.base import BaseDemoGenerator


class DemoRegistry:
    """Registry for demo data generators.

    Provides registration and lookup of BaseDemoGenerator subclasses by name.
    The global `demo_registry` instance is used by `generate_all_demos()` to
    discover and run all registered generators.
    """

    def __init__(self):
        self._generators: dict[DemoFeature, type[BaseDemoGenerator]] = {}

    def register(self, generator_cls: type[BaseDemoGenerator]) -> None:
        name = generator_cls.name
        if not name:
            raise ValueError(f"{generator_cls.__name__} must define 'name' class attribute")
        if name in self._generators:
            raise ValueError(f"Generator '{name}' is already registered")
        self._generators[name] = generator_cls

    def get(self, name: DemoFeature) -> type[BaseDemoGenerator]:
        if name not in self._generators:
            available = list(self._generators.keys())
            raise ValueError(f"Generator '{name}' not found. Available: {available}")
        return self._generators[name]

    def list_generators(self) -> list[DemoFeature]:
        return list(self._generators.keys())

    def __contains__(self, name: DemoFeature) -> bool:
        return name in self._generators


demo_registry = DemoRegistry()
