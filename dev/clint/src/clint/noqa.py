from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass
from typing import Iterator

NOQA_REGEX = re.compile(r"#\s*noqa(?:\s*:\s*([A-Z]\d+(?:\s*,\s*[A-Z]\d+)*))", re.IGNORECASE)


@dataclass
class Noqa:
    lineno: int
    col_offset: int
    rules: set[str]


def iter_noqa_comments(code: str) -> Iterator[Noqa]:
    readline = io.StringIO(code).readline
    tokens = tokenize.generate_tokens(readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue

            match = NOQA_REGEX.search(token.string.strip())
            if not match:
                continue

            rules_str = match.group(1)
            yield Noqa(
                lineno=token.start[0],
                col_offset=token.start[1],
                rules={r.strip() for r in rules_str.split(",")},
            )
    except tokenize.TokenError:
        # Handle incomplete tokens at end of file
        pass
