import sys
import mlflow  # pylint: disable=unused-import

cached_packages = set(k.split(".")[0] for k in sys.modules.keys())
assert "pyspark" not in cached_packages

# Make sure `pyspark` can be imported
import pyspark  # pylint: disable=unused-import
