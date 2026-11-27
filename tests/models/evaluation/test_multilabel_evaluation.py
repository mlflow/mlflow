import numpy as np
import pandas as pd

from mlflow.data.evaluation_dataset import _series_of_arrays_to_2d
from mlflow.models.evaluation.evaluators.classifier import _is_categorical


def _make_multilabel_df():
    # Two samples, 3 labels each (indicator vectors)
    return pd.DataFrame(
        {
            "y_true": [np.array([1, 0, 1]), np.array([0, 1, 1])],
            "y_pred": [np.array([1, 0, 1]), np.array([0, 1, 0])],
        }
    )


# -----------------------------
# Unit: _series_of_arrays_to_2d
# -----------------------------
def test_series_of_arrays_to_2d_converts_to_stacked_matrix():
    df = _make_multilabel_df()
    s = df["y_true"]  # Series of arrays
    arr = _series_of_arrays_to_2d(s)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == (2, 3)
    # Check contents
    np.testing.assert_array_equal(arr[0], np.array([1, 0, 1]))
    np.testing.assert_array_equal(arr[1], np.array([0, 1, 1]))


# -----------------------------
# Unit: _is_categorical for 2-D
# -----------------------------
def test_is_categorical_accepts_multilabel_indicator_matrix():
    y = np.array([[1, 0, 1], [0, 1, 1]])
    assert _is_categorical(y) is True
