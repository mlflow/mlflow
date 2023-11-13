from functools import reduce
from typing import Dict, NamedTuple, Tuple


class Message(NamedTuple):
    id: str
    name: str
    message: str
    reason: str

    def to_dict(self) -> Dict[str, Tuple[str, str, str]]:
        return {self.id: (self.message, self.name, self.reason)}


def to_msgs(*messages: Message) -> Dict[str, Tuple[str, str, str]]:
    return reduce(lambda x, y: {**x, **y.to_dict()}, messages, {})


LAZY_BUILTIN_IMPORT = Message(
    id="W0007",
    name="lazy-builtin-import",
    message="Import built-in module(s) (%s) at the top of the file.",
    reason="There is no reason they should be imported inside a function.",
)

USELESS_ASSIGNMENT = Message(
    id="W0008",
    name="useless-assignment",
    message="Useless assignment. Use immediate return instead.",
    reason="For simplicity and readability",
)

ILLEGAL_DIRECT_IMPORT = Message(
    id="W0009",
    name="illegal-direct-import",
    message="Direct import of the package `%s` is not allowed. "
            "Use the wrapper module `%s` instead.",
    reason="For some packages, we provide wrapper module to enforce some pre/post conditions."
           "Direct import of the package is not allowed as it bypasses these conditions.",
)