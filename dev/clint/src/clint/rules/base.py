import inspect
import itertools
import re
from abc import ABC, abstractmethod
from typing import Any

_id_counter = itertools.count(start=1)
_CLASS_NAME_TO_RULE_NAME_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


class Rule(ABC):
    id: str
    name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only generate ID for concrete classes
        if not inspect.isabstract(cls):
            id_ = next(_id_counter)
            cls.id = f"MLF{id_:04d}"
            cls.name = _CLASS_NAME_TO_RULE_NAME_REGEX.sub("-", cls.__name__).lower()

    @abstractmethod
    def _message(self) -> str:
        """
        Return a message that explains this rule.
        """

    @property
    def message(self) -> str:
        return self._message()
