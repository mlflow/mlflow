import pytest

from mlflow.utils.string_utils import strip_prefix, strip_suffix


@pytest.mark.parametrize("original,prefix,expected", [
    ('smoketest', 'smoke', 'test'),
    ('', 'test', ''),
    ('', '', ''),
    ('test', '', 'test')
])
def test_strip_prefix(original, prefix, expected):
    assert strip_prefix(original, prefix) == expected


@pytest.mark.parametrize("original,suffix,expected", [
    ('smoketest', 'test', 'smoke'),
    ('', 'test', ''),
    ('', '', ''),
    ('test', '', 'test')
])
def test_strip_suffix(original, suffix, expected):
    assert strip_suffix(original, suffix) == expected
