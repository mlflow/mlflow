# This test file must be executed via:
# $ python tests/test_mlflow_lazily_imports_pspark

if __name__ == "__main__":
    import sys
    import mlflow  # pylint: disable=unused-import

    cached_packages = set(k.split(".")[0] for k in sys.modules.keys())
    assert "pyspark" not in cached_packages

    # Make sure `pyspark` can be imported
    import pyspark  # pylint: disable=unused-import
