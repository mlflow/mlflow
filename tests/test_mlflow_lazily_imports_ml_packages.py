# This test file must be executed via:
# $ python tests/test_mlflow_lazily_imports_pspark

if __name__ == "__main__":
    import sys
    import mlflow  # pylint: disable=unused-import

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
        "shap",
        "sklearn",
        "spacy",
        "statsmodels",
        "tensorflow",
        "torch",
        "xgboost",
    }

    loaded_packages = {k.split(".")[0] for k in sys.modules.keys()}
    assert loaded_packages.intersection(ml_packages) == set()
