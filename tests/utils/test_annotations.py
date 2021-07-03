import pytest
from unittest import mock

from mlflow.utils.annotations import deprecate_conda_env


def _positional_arg(a, conda_env):
    """positional"""


def _keyword_arg(a, conda_env=None):
    """keyword"""


def keyword_only_arg(a, *, conda_env=None):
    """keyword_only"""




@pytest.mark.parametrize("original", [_positional_arg, _keyword_arg, keyword_only_arg])
def test_deprecate_conda_env(original):
    wrapped = deprecate_conda_env(original)
    assert wrapped.__name__ == original.__name__
    assert wrapped.__doc__ == original.__doc__

    if original != keyword_only_arg:
        with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
            wrapped(0, {})

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        wrapped(0, conda_env={})

    with pytest.warns(FutureWarning, match="`conda_env` has been deprecated"):
        wrapped(0, conda_env=None)

    if original != _positional_arg:
        with mock.patch("warnings.warn") as mock_warn:
            wrapped(0)
            mock_warn.assert_not_called()
