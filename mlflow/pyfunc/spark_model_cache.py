import os
import shutil
import tempfile
import zipfile

from pyspark.files import SparkFiles


class SparkModelCache(object):
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
    def add_local_model(spark, model_path):
        """Given a SparkSession and a model_path which refers to a pyfunc directory locally,
        we will zip the directory up, enable it to be distributed to executors, and return
        the "archive_path", which should be used as the path in get_or_load().
        """
        _, archive_basepath = tempfile.mkstemp()
        # NB: We must archive the directory as Spark.addFile does not support non-DFS
        # directories when recursive=True.
        archive_path = shutil.make_archive(archive_basepath, 'zip', model_path)
        spark.sparkContext.addFile(archive_path)
        return archive_path

    @staticmethod
    def get_or_load(archive_path):
        """Given a path returned by add_local_model(), this method will return the loaded model.
        If this Python process ever loaded the model before, we will reuse that copy.
        """
        if archive_path in SparkModelCache._models:
            SparkModelCache._cache_hits += 1
            return SparkModelCache._models[archive_path]

        # BUG: Despite the documentation of SparkContext.addFile() and SparkFiles.get() in Scala
        # and Python, it turns out that we actually need to use the basename as the input to
        # SparkFiles.get(), as opposed to the (absolute) path.
        archive_path_basename = os.path.basename(archive_path)
        local_path = SparkFiles.get(archive_path_basename)
        temp_dir = tempfile.mkdtemp()
        zip_ref = zipfile.ZipFile(local_path, 'r')
        zip_ref.extractall(temp_dir)
        zip_ref.close()

        # We must rely on a supposed cyclic import here because we want this behavior
        # on the Spark Executors (i.e., don't try to pickle the load_model function).
        from mlflow.pyfunc import load_pyfunc  # pylint: disable=cyclic-import
        SparkModelCache._models[archive_path] = load_pyfunc(temp_dir)
        return SparkModelCache._models[archive_path]
