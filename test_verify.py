"""Test module with single-line docstring."""


def test_sample():
    """Single-line docstring."""


class TestSample:
    """Another single-line docstring."""


def test_multiline():
    """
    This is a multiline docstring.
    It should not be flagged.
    """
