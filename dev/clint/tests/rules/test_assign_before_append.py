from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Position, Range, lint_file
from clint.rules import AssignBeforeAppend


def test_assign_before_append_basic(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item = transform(x)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert all(isinstance(r.rule, AssignBeforeAppend) for r in results)
    assert results[0].range == Range(Position(2, 0))


def test_assign_before_append_no_flag_different_variable(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item = transform(x)
    items.append(other_var)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_no_empty_list_init(index: SymbolIndex) -> None:
    code = """
for x in data:
    item = transform(x)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_different_list(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item = transform(x)
    other_list.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_three_statements(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item = transform(x)
    print(item)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_one_statement(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    items.append(transform(x))
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_list_with_initial_values(index: SymbolIndex) -> None:
    code = """
items = [1, 2, 3]
for x in data:
    item = transform(x)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_multiple_violations(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item = transform(x)
    items.append(item)

results = []
for y in other_data:
    result = process(y)
    results.append(result)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 2
    assert all(isinstance(r.rule, AssignBeforeAppend) for r in results)
    assert results[0].range == Range(Position(2, 0))
    assert results[1].range == Range(Position(7, 0))


def test_assign_before_append_no_flag_complex_assignment(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    item, other = transform(x)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_no_flag_attribute_assignment(index: SymbolIndex) -> None:
    code = """
items = []
for x in data:
    self.item = transform(x)
    items.append(self.item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_assign_before_append_separated_statements(index: SymbolIndex) -> None:
    code = """
items = []
other_statement()
for x in data:
    item = transform(x)
    items.append(item)
"""
    config = Config(select={AssignBeforeAppend.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0
