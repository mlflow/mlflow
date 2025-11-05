import os

import pytest


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_autolog_without_scipy():
    import mlflow  # clint: disable=package-import-in-test

    with pytest.raises(ImportError, match="scipy"):
        import scipy  # noqa: F401  # clint: disable=package-import-in-test

    assert not mlflow.models.utils.HAS_SCIPY

    mlflow.autolog()
    mlflow.models.utils._Example({})
