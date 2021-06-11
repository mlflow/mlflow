# If test test script is run after other test script that imports ML packages in the same pytest
# session, this test script fails. As a workaround, this test script needs to be run in a separate
# pytest session with the following command:
# $ pytest tests/test_mlflow_lazily_imports_ml_packages.py --lazy-import
import sys

import pytest


def _get_loaded_packages():
    return {k.split(".")[0] for k in sys.modules.keys()}


# Run this test only when the `--lazy-import` flag is specified
@pytest.mark.lazy_import
def test_mlflow_lazily_import_ml_packages():
    ml_packages = {
        "catboost",
        "fastai",
        "gluon",
        "h2o",
        "keras",
        "lightgbm",
        "mleap",
        "onnx",
        "pytorch_lightning",
        "pyspark",
        "shap",
        "sklearn",
        "spacy",
        "statsmodels",
        "tensorflow",
        "torch",
        "xgboost",
    }

    # Ensure ML packages are not loaded before importing mlflow
    assert _get_loaded_packages().intersection(ml_packages) == set()

    import mlflow  # pylint: disable=unused-import

    # Ensure ML packages are not loaded after importing mlflow
    assert _get_loaded_packages().intersection(ml_packages) == set()
