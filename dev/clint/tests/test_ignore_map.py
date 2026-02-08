from clint.linter import ignore_map


def test_ignore_map_single_rule() -> None:
    code = """
x = 1  # clint: disable=rule-a
y = 2
"""
    mapping = ignore_map(code)
    assert mapping == {"rule-a": {1}}


def test_ignore_map_multiple_rules() -> None:
    code = """
x = 1  # clint: disable=rule-a,rule-b
y = 2
"""
    mapping = ignore_map(code)
    assert mapping == {"rule-a": {1}, "rule-b": {1}}


def test_ignore_map_multiple_rules_with_spaces() -> None:
    code = """
x = 1  # clint: disable=rule-a, rule-b, rule-c
y = 2
"""
    mapping = ignore_map(code)
    assert mapping == {"rule-a": {1}, "rule-b": {1}, "rule-c": {1}}


def test_ignore_map_multiple_lines() -> None:
    code = """
x = 1  # clint: disable=rule-a
y = 2  # clint: disable=rule-b
z = 3  # clint: disable=rule-a,rule-b
"""
    mapping = ignore_map(code)
    assert mapping == {"rule-a": {1, 3}, "rule-b": {2, 3}}


def test_ignore_map_no_disable_comments() -> None:
    code = """
x = 1
y = 2
"""
    mapping = ignore_map(code)
    assert mapping == {}


def test_ignore_map_mixed_spacing() -> None:
    code = """
a = 1  # clint: disable=rule-a,rule-b
b = 2  # clint: disable=rule-c, rule-d
c = 3  # clint: disable=rule-e ,rule-f
d = 4  # clint: disable=rule-g , rule-h
"""
    mapping = ignore_map(code)
    assert mapping == {
        "rule-a": {1},
        "rule-b": {1},
        "rule-c": {2},
        "rule-d": {2},
        "rule-e": {3},
        "rule-f": {3},
        "rule-g": {4},
        "rule-h": {4},
    }
