import pytest

# Skip all tests in this directory if deepeval is not installed
pytest.importorskip("deepeval", reason="deepeval is not installed")
