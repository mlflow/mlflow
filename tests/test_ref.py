import os

import pytest

from mlflow.ref import REF


@pytest.mark.skipif("CI" not in os.environ, reason="Only works in CI")
def test_ref():
    assert REF is not None
