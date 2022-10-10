import pytest

from mlflow.utils.string_utils import strip_prefix, strip_suffix, is_string_type, mslex_quote


@pytest.mark.parametrize(
    (("original", "prefix", "expected")),
    [("smoketest", "smoke", "test"), ("", "test", ""), ("", "", ""), ("test", "", "test")],
)
def test_strip_prefix(original, prefix, expected):
    assert strip_prefix(original, prefix) == expected


@pytest.mark.parametrize(
    ("original", "suffix", "expected"),
    [("smoketest", "test", "smoke"), ("", "test", ""), ("", "", ""), ("test", "", "test")],
)
def test_strip_suffix(original, suffix, expected):
    assert strip_suffix(original, suffix) == expected


def test_is_string_type():
    assert is_string_type("validstring")
    assert is_string_type("")
    assert is_string_type((b"dog").decode("utf-8"))
    assert not is_string_type(None)
    assert not is_string_type(["teststring"])
    assert not is_string_type([])
    assert not is_string_type({})
    assert not is_string_type({"test": "string"})
    assert not is_string_type(12)
    assert not is_string_type(12.7)


def test_mslex_quote():
    assert mslex_quote("abc") == "abc"
    assert mslex_quote("a b c") == '"a b c"'
    assert mslex_quote("C:\\path\\to\\file") == "C:\\path\\to\\file"
