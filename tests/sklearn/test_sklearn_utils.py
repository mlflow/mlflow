import sys
from unittest.mock import patch
from mlflow.sklearn.utils import _all_estimators


def find_importer(module_name, attribute_name):
    for name, module in list(sys.modules.items()):
        if hasattr(module, attribute_name) and getattr(module, attribute_name).__module__ == module_name:
            return True
    return False


@patch('sklearn.utils.all_estimators', side_effect=ImportError)
def test_sklearn_utils_backported_all_estimators(mock_all_estimators):
    _all_estimators()
    _testing_check = find_importer('sklearn.utils._testing', 'ignore_warnings')
    assert _testing_check == True

    testing_check = find_importer('sklearn.utils.testing', 'ignore_warnings')
    assert testing_check == False
