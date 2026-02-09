from pathlib import Path

from dev.check_lazy_imports import compare_lazy_imports, extract_lazy_imports


def test_no_lazy_imports():
    code = """\
import os
from pathlib import Path
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_import_inside_function():
    code = """\
def foo():
    import json
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "json") in result
    assert result[("foo", "json")].line == 2


def test_from_import_inside_function():
    code = """\
def foo():
    from pathlib import Path
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "pathlib") in result


def test_multiple_lazy_imports():
    code = """\
def foo():
    import json
    from pathlib import Path
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("foo", "json") in result
    assert ("foo", "pathlib") in result


def test_type_checking_excluded():
    code = """\
from typing import TYPE_CHECKING

def foo():
    if TYPE_CHECKING:
        import json
    from pathlib import Path
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "pathlib") in result
    assert ("foo", "json") not in result


def test_typing_dot_type_checking_excluded():
    code = """\
import typing

def foo():
    if typing.TYPE_CHECKING:
        import json
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_type_checking_else_branch_not_excluded():
    code = """\
from typing import TYPE_CHECKING

def foo():
    if TYPE_CHECKING:
        import json
    else:
        import orjson
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "orjson") in result


def test_nested_function():
    code = """\
def outer():
    def inner():
        import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("outer.inner", "os") in result


def test_class_method():
    code = """\
class MyClass:
    def method(self):
        import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("MyClass.method", "os") in result


def test_nested_class_method():
    code = """\
def outer():
    class Inner:
        def method(self):
            import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("outer.Inner.method", "os") in result


def test_async_function():
    code = """\
async def fetch():
    import aiohttp
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("fetch", "aiohttp") in result


def test_import_alias():
    code = """\
def foo():
    import numpy as np
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "numpy") in result


def test_multiple_imports_one_statement():
    code = """\
def foo():
    import os, sys
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("foo", "os") in result
    assert ("foo", "sys") in result


def test_relative_import():
    code = """\
def foo():
    from . import utils
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", ".") in result


def test_relative_dotted_import():
    code = """\
def foo():
    from ..bar import baz
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "..bar") in result


def test_dotted_module_import():
    code = """\
def foo():
    import os.path
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "os.path") in result


def test_class_body_import_not_lazy():
    code = """\
class MyClass:
    import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 0


def test_same_module_in_different_functions():
    code = """\
def foo():
    import os

def bar():
    import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 2
    assert ("foo", "os") in result
    assert ("bar", "os") in result


def test_duplicate_import_in_same_function():
    code = """\
def foo():
    import os
    import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert result[("foo", "os")].line == 2


def test_deeply_nested_import():
    code = """\
def foo():
    if True:
        for x in []:
            import os
"""
    result = extract_lazy_imports(code)

    assert len(result) == 1
    assert ("foo", "os") in result


# --- compare_lazy_imports (base vs head diff) ---

FILE_PATH = Path("mlflow/example.py")


def test_compare_new_lazy_import_flagged():
    base = """\
def foo():
    pass
"""
    head = """\
def foo():
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'json'" in warnings[0].message


def test_compare_existing_lazy_import_not_flagged():
    base = """\
def foo():
    import json
"""
    head = """\
def foo():
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_removed_lazy_import_not_flagged():
    base = """\
def foo():
    import json
    import os
"""
    head = """\
def foo():
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_new_file_all_lazy_imports_flagged():
    head = """\
def foo():
    import json
    import os
"""
    warnings = compare_lazy_imports(FILE_PATH, None, head)

    assert len(warnings) == 2
    modules = {w.message.split("'")[1] for w in warnings}
    assert modules == {"json", "os"}


def test_compare_mixed_new_and_existing():
    base = """\
def foo():
    import json
"""
    head = """\
def foo():
    import json
    import os
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'os'" in warnings[0].message


def test_compare_new_function_with_lazy_import():
    base = """\
def foo():
    import json
"""
    head = """\
def foo():
    import json

def bar():
    import os
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert "'os'" in warnings[0].message


def test_compare_lazy_import_moved_to_top_level_not_flagged():
    base = """\
def foo():
    import json
"""
    head = """\
import json

def foo():
    pass
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 0


def test_compare_warning_has_correct_location():
    base = """\
def foo():
    pass
"""
    head = """\
def foo():
    x = 1
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    assert len(warnings) == 1
    assert warnings[0].file_path == FILE_PATH
    assert warnings[0].line == 3
    assert warnings[0].column == 5


def test_compare_warning_format_local():
    base = """\
def foo():
    pass
"""
    head = """\
def foo():
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    formatted = warnings[0].format(github=False)
    assert formatted.startswith("mlflow/example.py:2:5:")
    assert "[Non-blocking]" in formatted


def test_compare_warning_format_github():
    base = """\
def foo():
    pass
"""
    head = """\
def foo():
    import json
"""
    warnings = compare_lazy_imports(FILE_PATH, base, head)

    formatted = warnings[0].format(github=True)
    assert formatted.startswith("::warning file=mlflow/example.py,")
    assert "[Non-blocking]" in formatted
