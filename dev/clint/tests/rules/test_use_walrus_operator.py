from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import UseWalrusOperator


def test_basic_walrus_pattern(index_path: Path) -> None:
    code = """
a = func()
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)
    assert results[0].range == Range(Position(1, 0))


def test_walrus_in_function(index_path: Path) -> None:
    code = """
def foo():
    a = func()
    if a:
        use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_walrus_in_module(index_path: Path) -> None:
    code = """
result = compute()
if result:
    process(result)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UseWalrusOperator)


def test_no_flag_with_elif(index_path: Path) -> None:
    code = """
a = func()
if a:
    use(a)
elif other:
    do_other()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_with_else(index_path: Path) -> None:
    code = """
a = func()
if a:
    use(a)
else:
    do_other()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_variable_used_after_if(index_path: Path) -> None:
    code = """
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
a = func()
if a:
    do_something_else()
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_comparison_in_if(index_path: Path) -> None:
    code = """
a = func()
if a > 5:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_different_variable_in_if(index_path: Path) -> None:
    code = """
a = func()
if b:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_tuple_unpacking(index_path: Path) -> None:
    code = """
a, b = func()
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_multiple_targets(index_path: Path) -> None:
    code = """
a = b = func()
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_attribute_assignment(index_path: Path) -> None:
    code = """
self.a = func()
if self.a:
    use(self.a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_no_flag_multiline_assignment(index_path: Path) -> None:
    code = """
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
a: int = func()
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_multiple_violations(index_path: Path) -> None:
    code = """
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
a = func()
if a:
    def inner():
        return a
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Should still flag because nested function usage shouldn't count
    assert len(results) == 1


def test_no_flag_line_too_long(index_path: Path) -> None:
    # Create a very long function call that would make the line too long
    long_value = (
        "very_long_function_name_that_makes_the_line_exceed_one_hundred_"
        "characters_when_combined_with_walrus()"
    )
    code = f"""
a = {long_value}
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_flag_when_line_length_ok(index_path: Path) -> None:
    code = """
a = short()
if a:
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1


def test_no_flag_non_adjacent_statements(index_path: Path) -> None:
    code = """
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
a = func()
if a:
    class Inner:
        a = 5
    use(a)
"""
    config = Config(select={UseWalrusOperator.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    # Should still flag because class-level 'a' is a different scope
    assert len(results) == 1
