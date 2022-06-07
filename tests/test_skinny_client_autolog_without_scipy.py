import pytest
import os


class Scipy:
    class sparse:
        class csc_matrix:
            pass

        class csr_matrix:
            pass


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_autolog_without_scipy():
    import mlflow
    import numpy as np

    with pytest.raises(ImportError, match="scipy"):
        import scipy  # pylint: disable=unused-import

    assert not mlflow.models.utils.HAS_SCIPY

    mlflow.autolog()
    mlflow.models.utils._Example(dict())
