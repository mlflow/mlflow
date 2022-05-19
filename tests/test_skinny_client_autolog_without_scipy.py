import pytest
import os


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_autolog_without_scipy():
    import mlflow

    with pytest.raises(ImportError, match="scipy"):
        import scipy  # pylint: disable=unused-import

        assert scipy is not None  # pylint or flake8 disabling is not working

    mlflow.autolog()
