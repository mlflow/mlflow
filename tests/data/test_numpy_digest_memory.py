import tracemalloc

import numpy as np
import pandas as pd
import pytest

from mlflow.data.digest_utils import MAX_ROWS, compute_numpy_digest, get_normalized_md5_digest


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "reversed"])
def test_numpy_digest_allocates_only_for_the_hashed_prefix(layout):
    values = np.arange(2_000_000, dtype=np.float64).reshape(2000, 1000)
    if layout == "transposed":
        values = values.T
    elif layout == "reversed":
        values = values[::-1, ::-1]
    # Warm imports and pandas hashing before measuring the digest operation.
    compute_numpy_digest(np.arange(10, dtype=np.float64))

    tracemalloc.start()
    try:
        compute_numpy_digest(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The 80 KB sampled prefix and its hash buffers must not copy the 16 MB input.
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize(
    "values",
    [
        np.arange(20_000).reshape(100, 200),
        np.asfortranarray(np.arange(20_000).reshape(100, 200)),
        np.arange(20_000).reshape(100, 200).T,
        np.arange(20_000).reshape(100, 200)[::-1, ::-1],
        np.array(["first", "second", None], dtype=object),
        np.empty((0, 3)),
        np.array(7),
    ],
    ids=["contiguous", "fortran", "transposed", "reversed", "object", "empty", "scalar"],
)
def test_numpy_digest_preserves_existing_c_order_fingerprints(values):
    old_prefix = values.flatten()[:MAX_ROWS]
    expected = get_normalized_md5_digest(
        [pd.util.hash_array(old_prefix)] + [np.int64(size) for size in values.shape]
    )

    assert compute_numpy_digest(values) == expected


def test_numpy_digest_preserves_named_features_and_targets():
    features = {"z": np.arange(20_000).reshape(100, 200).T, "a": np.array([1.0, 2.0])}
    targets = {"label": np.arange(20_000)[::-1]}
    old_elements = []
    for arrays in (features, targets):
        for key in sorted(arrays):
            values = arrays[key]
            old_elements.extend([key.encode(), pd.util.hash_array(values.flatten()[:MAX_ROWS])])
            old_elements.extend(np.int64(size) for size in values.shape)

    assert compute_numpy_digest(features, targets) == get_normalized_md5_digest(old_elements)
