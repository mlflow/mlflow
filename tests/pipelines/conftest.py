import os
from unittest import mock
import pytest


@pytest.fixture(autouse=True)
def patch_get_pool_size():
    # Set the pool_size config for pandas_profiling to 1 on Windows to avoid using multiprocessing
    if os.name == "nt":
        with mock.patch("mlflow.pipelines.utils.step._get_pool_size", return_value=1):
            yield
    else:
        yield
