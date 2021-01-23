import pytest
import os


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_fails_import_flask():
    import mlflow

    assert mlflow is not None

    with pytest.raises(ImportError):
        import flask

        assert flask is not None


def test_fails_import_pandas():
    import mlflow

    assert mlflow is not None

    with pytest.raises(ImportError):
        import pandas

        assert pandas is not None


def test_fails_import_numpy():
    import mlflow

    assert mlflow is not None

    with pytest.raises(ImportError):
        import numpy

        assert numpy is not None
