import pytest
import os


@pytest.mark.skipif()
@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_fails_import_sqlalchemy():
    import mlflow

    assert mlflow is not None  # pylint or flake8 disabling is not working
    with pytest.raises(ImportError):
        import sqlalchemy  # pylint: disable=unused-import

        assert sqlalchemy is not None  # pylint or flake8 disabling is not working


def test_fails_import_flask():
    import mlflow

    assert mlflow is not None
    with pytest.raises(ImportError):
        import flask

        assert flask is not None
