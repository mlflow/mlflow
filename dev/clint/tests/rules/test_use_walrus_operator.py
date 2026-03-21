from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import UseWalrusOperator


def test_basic_walrus_pattern(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)
    assert results[0].range == Range(Position(2, 4))


def test_walrus_in_function(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_no_flag_walrus_in_module(index_path: Path) -> None:
    code = """
result = compute()
if result:
    process(result)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Module-level check is disabled for performance reasons
    assert len(results) == 0


def test_flag_with_elif_not_using_var(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
    elif other:
        do_other()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Flagged because var is not used in elif branch
    assert len(results) == 1


def test_no_flag_with_elif_using_var(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
    elif other:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Not flagged because var is used in elif branch
    assert len(results) == 0


def test_flag_with_else_not_using_var(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
    else:
        do_other()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Flagged because var is not used in else branch
    assert len(results) == 1


def test_no_flag_with_else_using_var(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
    else:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Not flagged because var is used in else branch
    assert len(results) == 0


def test_no_flag_variable_used_after_if(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
    print(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_variable_not_used_in_if_body(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        do_something_else()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_comparison_in_if(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a > 5:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_different_variable_in_if(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if b:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_tuple_unpacking(index_path: Path) -> None:
    code = """
def f():
    a, b = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_multiple_targets(index_path: Path) -> None:
    code = """
def f():
    a = b = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_attribute_assignment(index_path: Path) -> None:
    code = """
def f():
    self.a = func()
    if self.a:
        use(self.a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_multiline_assignment(index_path: Path) -> None:
    code = """
def f():
    a = (
        func()
    )
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_augmented_assignment(index_path: Path) -> None:
    code = """
def f():
    a = 1
    a += func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_annotated_assignment(index_path: Path) -> None:
    code = """
def f():
    a: int = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_multiple_violations(index_path: Path) -> None:
    code = """
def f():
    a = func1()
    if a:
        use(a)

    b = func2()
    if b:
        use(b)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, UseWalrusOperator) for r in results)


def test_nested_function_scope_not_considered(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        def inner():
            return a
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Flagged (false positive) - nested scopes are not handled for simplicity
    assert len(results) == 1


def test_no_flag_line_too_long(index_path: Path) -> None:
    long_value = (
        "very_long_function_name_that_makes_the_line_exceed_one_hundred_"
        "characters_when_combined_with_walrus()"
    )
    code = f"""
def f():
    a = {long_value}
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_flag_when_line_length_ok(index_path: Path) -> None:
    code = """
def f():
    a = short()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1


def test_no_flag_non_adjacent_statements(index_path: Path) -> None:
    code = """
def f():
    a = func()
    other_statement()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_variable_used_multiple_times_in_if_body(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
        process(a)
        print(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1


def test_nested_if_in_body(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        use(a)
        if other:
            process(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1


def test_class_scope_not_confused(index_path: Path) -> None:
    code = """
def f():
    a = func()
    if a:
        class Inner:
            a = 5
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Flagged (false positive) - nested scopes are not handled for simplicity
    assert len(results) == 1


def test_walrus_in_nested_if(index_path: Path) -> None:
    code = """
def f():
    if condition:
        a = func()
        if a:
            use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_walrus_in_for_loop(index_path: Path) -> None:
    code = """
def f():
    for x in items:
        a = func()
        if a:
            use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_walrus_in_while_loop(index_path: Path) -> None:
    code = """
def f():
    while condition:
        a = func()
        if a:
            use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_walrus_in_with_block(index_path: Path) -> None:
    code = """
def f():
    with context:
        a = func()
        if a:
            use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_walrus_in_try_block(index_path: Path) -> None:
    code = """
def f():
    try:
        a = func()
        if a:
            use(a)
    except Exception:
        pass
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)
