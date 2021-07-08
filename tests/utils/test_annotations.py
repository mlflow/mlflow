import pytest
from unittest import mock

from mlflow.utils.annotations import deprecate_conda_env


def _keyword_arg(a, conda_env=None, b=0):
    """keyword"""


def _keyword_only_arg(a, *, conda_env=None, b=0):
    """keyword_only"""


def test_deprecate_conda_env_preserves_function_attributes():
    wrapped = deprecate_conda_env(_keyword_arg)
    assert wrapped.__name__ == _keyword_arg.__name__
    assert wrapped.__doc__ == _keyword_arg.__doc__


def test_deprecate_conda_env_keyword_arg():
    wrapped = deprecate_conda_env(_keyword_arg)

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        wrapped(0, {})

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        wrapped(0, conda_env={})

    with mock.patch("warnings.warn") as mock_warn:
        wrapped(0, None)
        mock_warn.assert_not_called()

        mock_warn.reset_mock()
        wrapped(0, conda_env=None)
        mock_warn.assert_not_called()


def test_deprecate_conda_env_keyword_only_arg():
    wrapped = deprecate_conda_env(_keyword_only_arg)

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        wrapped(0, conda_env={})

    with mock.patch("warnings.warn") as mock_warn:
        wrapped(0, conda_env=None)
        mock_warn.assert_not_called()
