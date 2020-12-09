import sys

import pytest


@pytest.mark.skipif(sys.version_info[:2] != (3, 5), reason="This test fails in Python != 3.5")
def test_deprecate_python():
    with pytest.warns(DeprecationWarning, match="Python 3.5 is deprecated"):
        import mlflow  # noqa: F401
