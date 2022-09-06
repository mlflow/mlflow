import re

import pytest
from mlflow.utils.annotations import deprecated


class MyClass:
    @deprecated()
    def method(self):
        """
        Returns 0
        """
        return 0


@deprecated()
def function():
    """
    Returns 1
    """
    return 1


def test_deprecated_method():
    msg = "``tests.utils.test_annotations.MyClass.method`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert MyClass().method() == 0
    assert msg in MyClass.method.__doc__


def test_deprecated_function():
    msg = "``tests.utils.test_annotations.function`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert function() == 1
    assert msg in function.__doc__
