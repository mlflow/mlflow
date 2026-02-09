from pathlib import Path

from dev.check_lazy_imports import compare_lazy_imports, extract_lazy_imports


def test_no_lazy_imports():
    code = """\
import foo
from bar import baz
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_import_inside_function():
    code = """\
def func():
    import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo") in result
    assert result[("func", "foo")].line == 2


def test_from_import_inside_function():
    code = """\
def func():
    from foo import bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo") in result


def test_multiple_lazy_imports():
    code = """\
def func():
    import foo
    from bar import baz
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("func", "foo") in result
    assert ("func", "bar") in result


def test_type_checking_excluded():
    code = """\
from typing import TYPE_CHECKING

def func():
    if TYPE_CHECKING:
        import foo
    from bar import baz
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "bar") in result
    assert ("func", "foo") not in result


def test_typing_dot_type_checking_excluded():
    code = """\
import typing

def func():
    if typing.TYPE_CHECKING:
        import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_non_typing_type_checking_not_excluded():
    code = """\
def func():
    if foo.TYPE_CHECKING:
        import bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "bar") in result


def test_type_checking_else_branch_not_excluded():
    code = """\
from typing import TYPE_CHECKING

def func():
    if TYPE_CHECKING:
        import foo
    else:
        import bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "bar") in result


def test_nested_function():
    code = """\
def outer():
    def inner():
        import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("outer.inner", "foo") in result


def test_class_method():
    code = """\
class MyClass:
    def method(self):
        import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("MyClass.method", "foo") in result


def test_nested_class_method():
    code = """\
def outer():
    class Inner:
        def method(self):
            import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("outer.Inner.method", "foo") in result


def test_async_function():
    code = """\
async def func():
    import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo") in result


def test_import_alias():
    code = """\
def func():
    import foo as f
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo") in result


def test_multiple_imports_one_statement():
    code = """\
def func():
    import foo, bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("func", "foo") in result
    assert ("func", "bar") in result


def test_relative_import():
    code = """\
def func():
    from . import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", ".") in result


def test_relative_dotted_import():
    code = """\
def func():
    from ..foo import bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "..foo") in result


def test_dotted_module_import():
    code = """\
def func():
    import foo.bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo.bar") in result


def test_class_body_import_not_lazy():
    code = """\
class MyClass:
    import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_same_module_in_different_functions():
    code = """\
def foo():
    import bar

def baz():
    import bar
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("foo", "bar") in result
    assert ("baz", "bar") in result


def test_duplicate_import_in_same_function():
    code = """\
def func():
    import foo
    import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert result[("func", "foo")].line == 2


def test_deeply_nested_import():
    code = """\
def func():
    if True:
        for x in []:
            import foo
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("func", "foo") in result


# --- compare_lazy_imports (base vs head diff) ---

FILE_PATH = Path("mlflow/example.py")


def test_compare_new_lazy_import_flagged():
    base = """\
def func():
    pass
"""
    head = """\
def func():
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'foo'" in warnings[0].message


def test_compare_existing_lazy_import_not_flagged():
    base = """\
def func():
    import foo
"""
    head = """\
def func():
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_removed_lazy_import_not_flagged():
    base = """\
def func():
    import foo
    import bar
"""
    head = """\
def func():
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_new_file_all_lazy_imports_flagged():
    head = """\
def func():
    import foo
    import bar
"""
    warnings = compare_lazy_imports(FILE_PATH, None, head)

    assert len(warnings) == 2
    modules = {w.message.split("'")[1] for w in warnings}
    assert modules == {"foo", "bar"}


def test_compare_mixed_new_and_existing():
    base = """\
def func():
    import foo
"""
    head = """\
def func():
    import foo
    import bar
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'bar'" in warnings[0].message


def test_compare_new_function_with_lazy_import():
    base = """\
def foo():
    import bar
"""
    head = """\
def foo():
    import bar

def baz():
    import qux
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'qux'" in warnings[0].message


def test_compare_lazy_import_moved_to_top_level_not_flagged():
    base = """\
def func():
    import foo
"""
    head = """\
import foo

def func():
    pass
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_warning_has_correct_location():
    base = """\
def func():
    pass
"""
    head = """\
def func():
    x = 1
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert warnings[0].file_path == FILE_PATH
    assert warnings[0].line == 3
    assert warnings[0].column == 5


def test_compare_warning_format_local():
    base = """\
def func():
    pass
"""
    head = """\
def func():
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    formatted = warnings[0].format(github=False)
    assert formatted.startswith("mlflow/example.py:2:5:")
    assert "[Non-blocking]" in formatted


def test_compare_warning_format_github():
    base = """\
def func():
    pass
"""
    head = """\
def func():
    import foo
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    formatted = warnings[0].format(github=True)
    assert formatted.startswith("::warning file=mlflow/example.py,")
    assert "[Non-blocking]" in formatted
