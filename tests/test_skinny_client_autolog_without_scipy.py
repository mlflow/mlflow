import os

import numpy as np
import pandas as pd
import pytest

from mlflow.models import infer_signature
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_autolog_without_scipy():
    import mlflow

    with pytest.raises(ImportError, match="scipy"):
        import scipy  # noqa: F401

    assert not mlflow.models.utils.HAS_SCIPY

    mlflow.autolog()
    mlflow.models.utils._Example({})


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_infer_signature_without_scipy():
    with pytest.raises(ImportError, match="scipy"):
        import scipy  # noqa: F401

    signature = infer_signature(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), np.array([0.5, 1.5]))
    assert signature.inputs == Schema([ColSpec(DataType.long, "a"), ColSpec(DataType.string, "b")])
    assert signature.outputs == Schema([TensorSpec(np.dtype("float64"), (-1,))])
