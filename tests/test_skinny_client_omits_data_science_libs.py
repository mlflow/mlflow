import os

import pytest


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_fails_import_flask():
    import mlflow  # noqa: F401

    with pytest.raises(ImportError, match="flask"):
        import flask  # noqa: F401


def test_fails_import_pandas():
    import mlflow  # noqa: F401

    with pytest.raises(ImportError, match="pandas"):
        import pandas  # noqa: F401


def test_fails_import_numpy():
    import mlflow  # noqa: F401

    with pytest.raises(ImportError, match="numpy"):
        import numpy  # noqa: F401
