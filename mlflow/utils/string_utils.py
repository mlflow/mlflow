import re
import shlex
from typing import List

from mlflow.utils.os import is_windows


def strip_prefix(original, prefix):
    if original.startswith(prefix):
        return original[len(prefix) :]
    return original


def strip_suffix(original, suffix):
    if original.endswith(suffix) and suffix != "":
        return original[: -len(suffix)]
    return original


def is_string_type(item):
    return isinstance(item, str)


def generate_feature_name_if_not_string(s):
    if isinstance(s, str):
        return s

    return f"feature_{s}"


def truncate_str_from_middle(s, max_length):
    assert max_length > 5
    if len(s) <= max_length:
        return s
    else:
        left_part_len = (max_length - 3) // 2
        right_part_len = max_length - 3 - left_part_len
        return f"{s[:left_part_len]}...{s[-right_part_len:]}"


def _create_table(
    rows: List[List[str]], headers: List[str], column_sep: str = " " * 2, min_column_width: int = 4
) -> str:
    """
    Creates a table from a list of rows and headers.

    Example
    =======
    >>> print(_create_table([["a", "b", "c"], ["d", "e", "f"]], ["x", "y", "z"]))
    x     y     z
    ----  ----  ----
    a     b     c
    d     e     f
    """
    column_widths = [
        max(len(max(col, key=len)), len(header) + 2, min_column_width)
        for col, header in zip(zip(*rows), headers)
    ]
    aligned_rows = [
        column_sep.join(header.ljust(width) for header, width in zip(headers, column_widths)),
        column_sep.join("-" * width for width in column_widths),
        *(
            column_sep.join(cell.ljust(width) for cell, width in zip(row, column_widths))
            for row in rows
        ),
    ]
    return "\n".join(aligned_rows)


# Source: https://github.com/smoofra/mslex/blob/3338c347324d52af619ba39cebfdf7cbf46fa51b/mslex.py#L89-L139
cmd_meta = r"([\"\^\&\|\<\>\(\)\%\!])"
cmd_meta_or_space = r"[\s\"\^\&\|\<\>\(\)\%\!]"
cmd_meta_inside_quotes = r"([\"\%\!])"


def mslex_quote(s, for_cmd=True):
    """
    Quote a string for use as a command line argument in DOS or Windows.

    On windows, before a command line argument becomes a char* in a
    program's argv, it must be parsed by both cmd.exe, and by
    CommandLineToArgvW.

    If for_cmd is true, then this will quote the string so it will
    be parsed correctly by cmd.exe and then by CommandLineToArgvW.

    If for_cmd is false, then this will quote the string so it will
    be parsed correctly when passed directly to CommandLineToArgvW.

    For some strings there is no way to quote them so they will
    parse correctly in both situations.
    """
    if not s:
        return '""'
    if not re.search(cmd_meta_or_space, s):
        return s
    if for_cmd and re.search(cmd_meta, s):
        if not re.search(cmd_meta_inside_quotes, s):
            m = re.search(r"\\+$", s)
            if m:
                return '"' + s + m.group() + '"'
            else:
                return '"' + s + '"'
        if not re.search(r"[\s\"]", s):
            return re.sub(cmd_meta, r"^\1", s)
        return re.sub(cmd_meta, r"^\1", mslex_quote(s, for_cmd=False))
    i = re.finditer(r"(\\*)(\"+)|(\\+)|([^\\\"]+)", s)

    def parts():
        yield '"'
        for m in i:
            _, end = m.span()
            slashes, quotes, onlyslashes, text = m.groups()
            if quotes:
                yield slashes
                yield slashes
                yield r"\"" * len(quotes)
            elif onlyslashes:
                if end == len(s):
                    yield onlyslashes
                    yield onlyslashes
                else:
                    yield onlyslashes
            else:
                yield text
        yield '"'

    return "".join(parts())


def quote(s):
    return mslex_quote(s) if is_windows() else shlex.quote(s)
