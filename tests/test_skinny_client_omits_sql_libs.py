import pytest
import os


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_fails_import_sqlalchemy():
    import mlflow

    assert mlflow is not None  # pylint or flake8 disabling is not working

    with pytest.raises(ImportError, match="sqlalchemy"):
        import sqlalchemy  # pylint: disable=unused-import

        assert sqlalchemy is not None  # pylint or flake8 disabling is not working
