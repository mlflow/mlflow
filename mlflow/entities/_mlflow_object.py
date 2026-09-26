import pprint
from abc import abstractmethod
from collections.abc import Iterator
from functools import cached_property
from typing import Any


class _MlflowObject:
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    def _get_properties_helper(cls):
        return sorted([
            p for p in cls.__dict__ if isinstance(getattr(cls, p), (property, cached_property))
        ])

    @classmethod
    def _properties(cls):
        return cls._get_properties_helper()

    @classmethod
    @abstractmethod
    def from_proto(cls, proto: Any) -> "_MlflowObject":
        pass

    @classmethod
    def from_dictionary(cls, the_dict: dict[str, Any]) -> "_MlflowObject":
        filtered_dict = {key: value for key, value in the_dict.items() if key in cls._properties()}
        return cls(**filtered_dict)

    def __repr__(self) -> str:
        return to_string(self)


def to_string(obj: Any) -> str:
    return _MlflowObjectPrinter().to_string(obj)


def get_classname(obj: Any) -> str:
    return type(obj).__name__


class _MlflowObjectPrinter:
    def __init__(self) -> None:
        super().__init__()
        self.printer = pprint.PrettyPrinter()

    def to_string(self, obj: Any) -> str:
        if isinstance(obj, _MlflowObject):
            return f"<{get_classname(obj)}: {self._entity_to_string(obj)}>"
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ", ".join([f"{key}={self.to_string(value)}" for key, value in entity])
