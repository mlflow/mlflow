import pytest
from unittest import mock

from mlflow.utils.annotations import deprecate_conda_env


def test_deprecate_conda_env():
    @deprecate_conda_env
    def f(a, conda_env=None):
        """docstring"""

    assert f.__name__ == "f"
    assert f.__doc__ == "docstring"

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        f(0, {})

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        f(0, conda_env={})

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        f(0, conda_env=None)

    with mock.patch("warnings.warn") as mock_warn:
        f(0)
        mock_warn.assert_not_called()
