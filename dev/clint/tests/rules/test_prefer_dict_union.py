from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import lint_file
from clint.rules import PreferDictUnion


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("{**dict1, **dict2}", id="two_dict_unpacks"),
        pytest.param("{**dict1, **dict2, **dict3}", id="three_dict_unpacks"),
        pytest.param("{**a, **b, **c, **d}", id="four_dict_unpacks"),
        pytest.param("{**obj.attr, **other}", id="attribute_and_name"),
        pytest.param("{**a.x, **b.y}", id="two_attributes"),
        pytest.param("{**a.b.c, **d}", id="chained_attribute"),
    ],
)
def test_flag(index_path: Path, code: str) -> None:
    config = Config(select={PreferDictUnion.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, PreferDictUnion)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("{**dict1}", id="single_dict_unpack"),
        pytest.param('{**dict1, "extra_key": "value"}', id="unpack_then_literal"),
        pytest.param('{"key": "value", **dict1}', id="literal_then_unpack"),
        pytest.param('{"key1": "value1", "key2": "value2"}', id="only_literals"),
        pytest.param("{}", id="empty_dict"),
        pytest.param('{**dict1, "k1": "v1", **dict2, "k2": "v2"}', id="mixed_unpacks_and_literals"),
        pytest.param('{**dict1, **dict2, "override": "value"}', id="unpacks_with_trailing_literal"),
        pytest.param("{**data[0], **other}", id="subscript_access"),
        pytest.param('{**configs, **{"key": "value"}}', id="dict_literal_unpack"),
        pytest.param("{**func(), **other}", id="function_call"),
        pytest.param("{**a,\n**b}", id="multi_line"),
    ],
)
def test_no_flag(index_path: Path, code: str) -> None:
    config = Config(select={PreferDictUnion.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0
