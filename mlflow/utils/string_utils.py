from typing import List


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
    rows: List[List[str]], headers: List[str], column_sep: str = "  ", min_column_width=4
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
