import pytest
import os


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_fails_import_flask():
    import mlflow  # pylint: disable=unused-import

    with pytest.raises(ImportError, match="flask"):
        import flask  # pylint: disable=unused-import


def test_fails_import_pandas():
    import mlflow  # pylint: disable=unused-import

    with pytest.raises(ImportError, match="pandas"):
        import pandas  # pylint: disable=unused-import


def test_fails_import_numpy():
    import mlflow  # pylint: disable=unused-import

    with pytest.raises(ImportError, match="numpy"):
        import numpy  # pylint: disable=unused-import
