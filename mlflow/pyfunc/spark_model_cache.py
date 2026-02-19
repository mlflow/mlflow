from mlflow.utils._spark_utils import _SparkDirectoryDistributor


class SparkModelCache:
    """Caches models in memory on Spark Executors, to avoid continually reloading from disk.

    This class has to be part of a different module than the one that _uses_ it. This is
    because Spark will pickle classes that are defined in the local scope, but relies on
    Python's module loading behavior for classes in different modules. In this case, we
    are relying on the fact that Python will load a module at-most-once, and can therefore
    store per-process state in a static map.
    """

    # Map from unique name --> (loaded model, local_model_path).
    _models = {}

    # Number of cache hits we've had, for testing purposes.
    _cache_hits = 0

    def __init__(self):
        pass

    @staticmethod
    def add_local_model(spark, model_path):
        """Given a SparkSession and a model_path which refers to a pyfunc directory locally,
        we will zip the directory up, enable it to be distributed to executors, and return
        the "archive_path", which should be used as the path in get_or_load().
        """
        return _SparkDirectoryDistributor.add_dir(spark, model_path)

    @staticmethod
    def get_or_load(archive_path):
        """Given a path returned by add_local_model(), this method will return a tuple of
        (loaded_model, local_model_path).
        If this Python process ever loaded the model before, we will reuse that copy.
        """
        if archive_path in SparkModelCache._models:
            SparkModelCache._cache_hits += 1
            return SparkModelCache._models[archive_path]

        local_model_dir = _SparkDirectoryDistributor.get_or_extract(archive_path)

        # We must rely on a supposed cyclic import here because we want this behavior
        # on the Spark Executors (i.e., don't try to pickle the load_model function).
        from mlflow.pyfunc import load_model

        SparkModelCache._models[archive_path] = (load_model(local_model_dir), local_model_dir)
        return SparkModelCache._models[archive_path]
