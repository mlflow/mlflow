import numpy as np
import pandas as pd
import pytest

from mlflow.data.digest_utils import (
    _DIGEST_SIZE,
    compute_numpy_digest,
    compute_pandas_digest,
    get_normalized_md5_digest,
)


class TestComputePandasDigest:
    def test_basic_digest_length(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        digest = compute_pandas_digest(df)
        assert len(digest) == _DIGEST_SIZE

    def test_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert compute_pandas_digest(df) == compute_pandas_digest(df)

    def test_different_data_different_digest(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_datetime_columns_included(self):
        """Non-string/non-numeric columns like datetime must affect the digest."""
        df1 = pd.DataFrame({
            "id": [1, 2],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        })
        df2 = pd.DataFrame({
            "id": [1, 2],
            "ts": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        })
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_bool_columns_included(self):
        df1 = pd.DataFrame({"flag": [True, False, True]})
        df2 = pd.DataFrame({"flag": [False, True, False]})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_category_columns_included(self):
        df1 = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])})
        df2 = pd.DataFrame({"cat": pd.Categorical(["b", "a", "b"])})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_dtype_difference_changes_digest(self):
        """int32 vs int64 columns with the same values must produce different digests."""
        df1 = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int32)})
        df2 = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_tail_rows_affect_digest(self):
        """Datasets differing only beyond MAX_ROWS must produce different digests."""
        n = 10_001
        base = list(range(n))
        df1 = pd.DataFrame({"a": base})
        modified = base.copy()
        modified[-1] = 999_999
        df2 = pd.DataFrame({"a": modified})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_column_name_affects_digest(self):
        df1 = pd.DataFrame({"col_a": [1, 2, 3]})
        df2 = pd.DataFrame({"col_b": [1, 2, 3]})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)

    def test_row_count_affects_digest(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3, 4]})
        assert compute_pandas_digest(df1) != compute_pandas_digest(df2)


class TestComputeNumpyDigest:
    def test_basic_digest_length(self):
        arr = np.array([1, 2, 3])
        digest = compute_numpy_digest(arr)
        assert len(digest) == _DIGEST_SIZE

    def test_deterministic(self):
        arr = np.array([1, 2, 3])
        assert compute_numpy_digest(arr) == compute_numpy_digest(arr)

    def test_different_data_different_digest(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert compute_numpy_digest(arr1) != compute_numpy_digest(arr2)

    def test_dtype_difference_changes_digest(self):
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert compute_numpy_digest(arr1) != compute_numpy_digest(arr2)

    def test_tail_elements_affect_digest(self):
        n = 10_001
        arr1 = np.arange(n, dtype=np.float64)
        arr2 = arr1.copy()
        arr2[-1] = 999_999.0
        assert compute_numpy_digest(arr1) != compute_numpy_digest(arr2)

    def test_with_targets(self):
        features = np.array([1, 2, 3])
        targets1 = np.array([0, 1, 0])
        targets2 = np.array([1, 0, 1])
        assert compute_numpy_digest(features, targets1) != compute_numpy_digest(features, targets2)

    def test_dict_features(self):
        f1 = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        f2 = {"a": np.array([1, 2]), "b": np.array([5, 6])}
        assert compute_numpy_digest(f1) != compute_numpy_digest(f2)

    def test_shape_affects_digest(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([1, 2, 3, 4])
        assert compute_numpy_digest(arr1) != compute_numpy_digest(arr2)


class TestGetNormalizedMd5Digest:
    def test_backward_compat_returns_correct_length(self):
        digest = get_normalized_md5_digest([b"test"])
        assert len(digest) == _DIGEST_SIZE

    def test_empty_elements_raises(self):
        with pytest.raises(Exception, match="No hashable elements"):
            get_normalized_md5_digest([])
