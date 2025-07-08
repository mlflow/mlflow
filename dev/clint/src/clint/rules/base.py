from __future__ import annotations

import inspect
import itertools
import re
from abc import ABC, abstractmethod


class Rule(ABC):
    _CLASS_NAME_TO_RULE_NAME_REGEX = re.compile(r"(?<!^)(?=[A-Z])")
    _id_counter = itertools.count(start=1)
    _generated_id: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only generate ID for concrete classes
        if not inspect.isabstract(cls):
            id_ = next(cls._id_counter)
            cls._generated_id = f"MLF{id_:04d}"

    @property
    def id(self) -> str:
        return self._generated_id

    @abstractmethod
    def _message(self) -> str:
        """
        Return a message that explains this rule.
        """

    @property
    def message(self) -> str:
        return self._message()

    @property
    def name(self) -> str:
        """
        The name of this rule.
        """
        return self._CLASS_NAME_TO_RULE_NAME_REGEX.sub("-", self.__class__.__name__).lower()
