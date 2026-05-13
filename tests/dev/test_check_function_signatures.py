import ast

from dev.check_function_signatures import check_signature_compatibility


def test_no_changes():
    old_code = "def func(a, b=1): pass"
    new_code = "def func(a, b=1): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 0


def test_positional_param_removed():
    old_code = "def func(a, b, c): pass"
    new_code = "def func(a, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "Positional param 'c' was removed."
    assert errors[0].param_name == "c"


def test_positional_param_renamed():
    old_code = "def func(a, b): pass"
    new_code = "def func(x, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert "Positional param order/name changed: 'a' -> 'x'." in errors[0].message
    assert errors[0].param_name == "x"


def test_only_first_positional_rename_flagged():
    old_code = "def func(a, b, c, d): pass"
    new_code = "def func(x, y, z, w): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert "Positional param order/name changed: 'a' -> 'x'." in errors[0].message


def test_optional_positional_became_required():
    old_code = "def func(a, b=1): pass"
    new_code = "def func(a, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "Optional positional param 'b' became required."
    assert errors[0].param_name == "b"


def test_multiple_optional_became_required():
    old_code = "def func(a, b=1, c=2): pass"
    new_code = "def func(a, b, c): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 2
    assert errors[0].message == "Optional positional param 'b' became required."
    assert errors[1].message == "Optional positional param 'c' became required."


def test_new_required_positional_param():
    old_code = "def func(a): pass"
    new_code = "def func(a, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "New required positional param 'b' added."
    assert errors[0].param_name == "b"


def test_new_optional_positional_param_allowed():
    old_code = "def func(a): pass"
    new_code = "def func(a, b=1): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 0


def test_keyword_only_param_removed():
    old_code = "def func(*, a, b): pass"
    new_code = "def func(*, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "Keyword-only param 'a' was removed."
    assert errors[0].param_name == "a"


def test_multiple_keyword_only_removed():
    old_code = "def func(*, a, b, c): pass"
    new_code = "def func(*, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 2
    error_messages = {e.message for e in errors}
    assert "Keyword-only param 'a' was removed." in error_messages
    assert "Keyword-only param 'c' was removed." in error_messages


def test_optional_keyword_only_became_required():
    old_code = "def func(*, a=1): pass"
    new_code = "def func(*, a): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "Keyword-only param 'a' became required."
    assert errors[0].param_name == "a"


def test_new_required_keyword_only_param():
    old_code = "def func(*, a): pass"
    new_code = "def func(*, a, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "New required keyword-only param 'b' added."
    assert errors[0].param_name == "b"


def test_new_optional_keyword_only_allowed():
    old_code = "def func(*, a): pass"
    new_code = "def func(*, a, b=1): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 0


def test_complex_mixed_violations():
    old_code = "def func(a, b=1, *, c, d=2): pass"
    new_code = "def func(x, b, *, c=3, e): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 3
    error_messages = [e.message for e in errors]
    assert any("Positional param order/name changed: 'a' -> 'x'." in msg for msg in error_messages)
    assert any("Keyword-only param 'd' was removed." in msg for msg in error_messages)
    assert any("New required keyword-only param 'e' added." in msg for msg in error_messages)


def test_parameter_error_has_location_info():
    old_code = "def func(a): pass"
    new_code = "def func(b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].lineno == 1
    assert errors[0].col_offset > 0


def test_async_function_compatibility():
    old_code = "async def func(a, b=1): pass"
    new_code = "async def func(a, b): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert errors[0].message == "Optional positional param 'b' became required."


def test_positional_only_compatibility():
    old_code = "def func(a, /): pass"
    new_code = "def func(b, /): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert "Positional param order/name changed: 'a' -> 'b'." in errors[0].message


def test_rename_stops_further_positional_checks():
    old_code = "def func(a, b=1, c=2): pass"
    new_code = "def func(x, b, c): pass"

    old_tree = ast.parse(old_code)
    new_tree = ast.parse(new_code)
    errors = check_signature_compatibility(old_tree.body[0], new_tree.body[0])

    assert len(errors) == 1
    assert "Positional param order/name changed: 'a' -> 'x'." in errors[0].message
