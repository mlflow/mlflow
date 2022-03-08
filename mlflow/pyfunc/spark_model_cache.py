import os
import shutil
import tempfile
import zipfile

from mlflow.utils._spark_utils import _SparkBroadcastFileCache


# TODO: for NFS available cases, use NFS instead of spark files for distributing model to remote workers.
class SparkModelCache:
    """Caches models in memory on Spark Executors, to avoid continually reloading from disk.

    This class has to be part of a different module than the one that _uses_ it. This is
    because Spark will pickle classes that are defined in the local scope, but relies on
    Python's module loading behavior for classes in different modules. In this case, we
    are relying on the fact that Python will load a module at-most-once, and can therefore
    store per-process state in a static map.
    """

    # Map from unique name --> loaded model.
    _models = {}

    # Number of cache hits we've had, for testing purposes.
    _cache_hits = 0

    def __init__(self):
        pass

    @staticmethod
    def add_local_model(model_path):
        """Given a model_path which refers to a pyfunc directory locally,
        we will zip the directory up, enable it to be distributed to executors.
        This method must be called from spark driver side.
        return a cache key for the model_path
        """
        model_path = os.path.normpath(model_path)
        _, archive_basepath = tempfile.mkstemp()
        return _SparkBroadcastFileCache.add_file(model_path)

    @staticmethod
    def get_or_load(cache_key):
        """
        Given a cache key returned by `add_local_model`, this method will return the loaded model.
        This method can be called from either spark UDF routine or driver side.
        """
        if cache_key in SparkModelCache._models:
            SparkModelCache._cache_hits += 1
            return SparkModelCache._models[cache_key]

        local_path = _SparkBroadcastFileCache.get_file(cache_key)

        # We must rely on a supposed cyclic import here because we want this behavior
        # on the Spark Executors (i.e., don't try to pickle the load_model function).
        from mlflow.pyfunc import _load_model_from_local_path  # pylint: disable=cyclic-import

        SparkModelCache._models[cache_key] = _load_model_from_local_path(local_path)
        return SparkModelCache._models[cache_key]
