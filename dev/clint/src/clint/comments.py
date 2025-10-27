import io
import re
import tokenize
from dataclasses import dataclass
from typing import Iterator

from typing_extensions import Self

NOQA_REGEX = re.compile(r"#\s*noqa\s*:\s*([A-Z]\d+(?:\s*,\s*[A-Z]\d+)*)", re.IGNORECASE)


@dataclass
class Noqa:
    lineno: int
    col_offset: int
    rules: set[str]

    @classmethod
    def from_token(cls, token: tokenize.TokenInfo) -> Self | None:
        if match := NOQA_REGEX.match(token.string):
            rules = set(match.group(1).upper().split(","))
            return cls(
                lineno=token.start[0],
                col_offset=token.start[1],
                rules=rules,
            )
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
