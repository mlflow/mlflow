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
