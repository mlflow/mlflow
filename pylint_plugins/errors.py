from typing import NamedTuple, Dict, Tuple
from functools import reduce


class Message(NamedTuple):
    id: str
    name: str
    message: str
    reason: str

    def to_dict(self) -> Dict[str, Tuple[str, str, str]]:
        return {self.id: (self.message, self.name, self.reason)}


def to_msgs(*messages: Message) -> Dict[str, Tuple[str, str, str]]:
    return reduce(lambda x, y: {**x, **y.to_dict()}, messages, {})


PYTEST_RAISES_WITHOUT_MATCH = Message(
    id="W0001",
    name="pytest-raises-without-match",
    message="`pytest.raises` must be called with `match` argument`.",
    reason="`pytest.raises` without `match` argument can lead to false positives.",
)


UNITTEST_PYTEST_RAISES = Message(
    id="W0003",
    name="unittest-assert-raises",
    message="Use `pytest.raises` instead of `unittest.TestCase.assertRaises`.",
    reason="To enforce 'pytest-raises-multiple-statements' Message.",
)

PYTEST_RAISES_MULTIPLE_STATEMENTS = Message(
    id="W0004",
    name="pytest-raises-multiple-statements",
    message=(
        "`with pytest.raises` should not contain multiple statements. "
        "It should only contain a single statement that throws an exception."
    ),
    reason=(
        "To prevent unreachable assertions and make it easier to tell which line is "
        "expected to throw."
    ),
)


USE_F_STRING = Message(
    id="W0006",
    name="use-f-string",
    message="Use f-string instead of format",
    reason='`f"{foo} bar"` is simpler and faster than `"{} bar".format(foo)`',
)

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
