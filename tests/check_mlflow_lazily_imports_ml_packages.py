import sys

import pytest


@pytest.mark.parametrize(
    "package",
    (
        "catboost",
        "fastai",
        "mxnet",
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
    ),
)
def test_mlflow_lazily_imports_ml_packages(package):
    cached_packages = list(sys.modules.keys())
    for pkg in cached_packages:
        if pkg.split(".")[0] == pkg:
            sys.modules.pop(package, None)

    # Ensure the package is not loaded
    assert package not in sys.modules

    import mlflow  # pylint: disable=unused-import

    # Ensure mlflow didn't load the package
    assert package not in sys.modules
