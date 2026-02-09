from clint.linter import DisableComment, parse_disable_comments


def test_single_rule() -> None:
    code = """
x = 1  # clint: disable=rule-a
y = 2
"""
    assert parse_disable_comments(code) == [DisableComment("rule-a", 1, 9)]


def test_multiple_rules() -> None:
    code = """
x = 1  # clint: disable=rule-a,rule-b
y = 2
"""
    assert parse_disable_comments(code) == [
        DisableComment("rule-a", 1, 9),
        DisableComment("rule-b", 1, 9),
    ]


def test_multiple_rules_with_spaces() -> None:
    code = """
x = 1  # clint: disable=rule-a, rule-b, rule-c
y = 2
"""
    assert parse_disable_comments(code) == [
        DisableComment("rule-a", 1, 9),
        DisableComment("rule-b", 1, 9),
        DisableComment("rule-c", 1, 9),
    ]


def test_multiple_lines() -> None:
    code = """
x = 1  # clint: disable=rule-a
y = 2  # clint: disable=rule-b
z = 3  # clint: disable=rule-a,rule-b
"""
    assert parse_disable_comments(code) == [
        DisableComment("rule-a", 1, 9),
        DisableComment("rule-b", 2, 9),
        DisableComment("rule-a", 3, 9),
        DisableComment("rule-b", 3, 9),
    ]


def test_no_disable_comments() -> None:
    code = """
x = 1
y = 2
"""
    assert parse_disable_comments(code) == []


def test_various_spacing_around_commas() -> None:
    code = """
a = 1  # clint: disable=rule-a,rule-b
b = 2  # clint: disable=rule-c, rule-d
c = 3  # clint: disable=rule-e ,rule-f
d = 4  # clint: disable=rule-g , rule-h
"""
    assert parse_disable_comments(code) == [
        DisableComment("rule-a", 1, 9),
        DisableComment("rule-b", 1, 9),
        DisableComment("rule-c", 2, 9),
        DisableComment("rule-d", 2, 9),
        DisableComment("rule-e", 3, 9),
        DisableComment("rule-f", 3, 9),
        DisableComment("rule-g", 4, 9),
        DisableComment("rule-h", 4, 9),
    ]
