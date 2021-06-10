# This test file must be executed via:
# $ python tests/test_mlflow_lazily_imports_pspark

if __name__ == "__main__":
    import sys
    import mlflow  # pylint: disable=unused-import

    pyspark_modules = [k for k in sys.modules.keys() if k.split(".")[0] == "pyspark"]
    assert pyspark_modules == []

    # Make sure `pyspark` can be imported
    import pyspark  # pylint: disable=unused-import
