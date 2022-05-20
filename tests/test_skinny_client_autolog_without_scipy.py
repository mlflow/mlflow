import pytest
import os

from unittest import mock


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

        assert scipy is not None  # pylint or flake8 disabling is not working

    assert not mlflow.models.utils.HAS_SCIPY

    mlflow.autolog()
    mlflow.models.utils._Example(dict())

    with mock.patch.multiple(
        'mlflow.models.utils',
        HAS_SCIPY=True,
    ):
        mlflow.models.utils.csc_matrix = Scipy.sparse.csc_matrix,
        mlflow.models.utils.csr_matrix = Scipy.sparse.csr_matrix

        csc = mock.MagicMock(spec=Scipy.sparse.csc_matrix)
        csc.data = np.array([1, 2, 3])
        csc.indices = np.array([1, 2, 3])
        csc.indptr = mock.MagicMock()
        csc.indptr.tolist = lambda: [1, 2, 3]
        csc.shape = (3, 3)
        mlflow.models.utils._Example(csc)
