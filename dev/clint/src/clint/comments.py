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
class Comment:
    string: str
    start: "Position"
    end: "Position"


@dataclass
class Noqa:
    start: "Position"
    end: "Position"
    rules: set[str]

    @classmethod
    def from_comment(cls, comment: Comment) -> Self | None:
        if match := NOQA_REGEX.match(comment.string):
            rules = {r.strip() for r in match.group(1).upper().split(",")}
            return cls(start=comment.start, end=comment.end, rules=rules)
        return None


def iter_comments(code: str) -> Iterator[Comment]:
    # Import here to avoid circular dependency
    from clint.linter import Position

    readline = io.StringIO(code).readline
    try:
        tokens = tokenize.generate_tokens(readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                yield Comment(
                    string=token.string,
                    start=Position(token.start[0], token.start[1]),
                    end=Position(token.end[0], token.end[1]),
                )

    except tokenize.TokenError:
        # Handle incomplete tokens at end of file
        pass
