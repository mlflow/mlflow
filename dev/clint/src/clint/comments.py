import io
import re
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from typing_extensions import Self

if TYPE_CHECKING:
    from clint.linter import Position

NOQA_REGEX = re.compile(r"#\s*noqa\s*:\s*([A-Z]\d+(?:\s*,\s*[A-Z]\d+)*)", re.IGNORECASE)


@dataclass
class Noqa:
    start: "Position"
    end: "Position"
    rules: set[str]

    @classmethod
    def from_token(cls, token: tokenize.TokenInfo) -> Self | None:
        # Import here to avoid circular dependency
        from clint.linter import Position

        if match := NOQA_REGEX.match(token.string):
            rules = set(match.group(1).upper().split(","))
            start = Position(token.start[0], token.start[1])
            end = Position(token.end[0], token.end[1])
            return cls(start=start, end=end, rules=rules)
        return None


def iter_comments(code: str) -> Iterator[tokenize.TokenInfo]:
    readline = io.StringIO(code).readline
    try:
        tokens = tokenize.generate_tokens(readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                yield token

    except tokenize.TokenError:
        # Handle incomplete tokens at end of file
        pass
