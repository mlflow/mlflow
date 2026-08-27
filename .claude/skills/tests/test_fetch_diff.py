import re

import pytest
from skills.commands.fetch_diff import filter_diff

DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -268,3 +268,4 @@ def f():
     first
+    added
     second
     third
"""


def new_line_numbers(rendered: str) -> list[int]:
    return [int(n) for n in re.findall(r"^\s*\d*\s+(\d+) \|", rendered, re.MULTILINE)]


@pytest.mark.parametrize("diff", [DIFF, DIFF.rstrip("\n"), DIFF + "\n\n"])
def test_numbering_stops_at_the_last_hunk_line(diff: str) -> None:
    # @@ -268,3 +268,4 @@ covers new lines 268..271; a trailing newline must not add 272.
    assert max(new_line_numbers(filter_diff(diff))) == 271


def test_a_trailing_newline_adds_no_extra_rendered_line() -> None:
    assert filter_diff(DIFF) == filter_diff(DIFF.rstrip("\n"))


def test_context_and_added_lines_keep_their_numbers() -> None:
    rendered = filter_diff(DIFF)
    assert "  268   268 |      first" in rendered
    assert "        269 | +    added" in rendered
    assert "  269   270 |      second" in rendered
