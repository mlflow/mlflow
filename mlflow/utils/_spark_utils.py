import tempfile
import shutil
import os


def _get_active_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        # Return None if user doesn't have PySpark installed
        return None
    try:
        # getActiveSession() only exists in Spark 3.0 and above
        return SparkSession.getActiveSession()
    except Exception:
        # Fall back to this internal field for Spark 2.x and below.
        return SparkSession._instantiatedSession


class _SparkDirectoryDistributor:
    """Distribute spark directory from driver to executors."""

    _extracted_dir_paths = {}

    def __init__(self):
        pass

    @staticmethod
    def add_dir(spark, dir_path):
        """Given a SparkSession and a model_path which refers to a pyfunc directory locally,
        we will zip the directory up, enable it to be distributed to executors, and return
        the "archive_path", which should be used as the path in get_or_load().
        """
        _, archive_basepath = tempfile.mkstemp()
        # NB: We must archive the directory as Spark.addFile does not support non-DFS
        # directories when recursive=True.
        archive_path = shutil.make_archive(archive_basepath, "zip", dir_path)
        spark.sparkContext.addFile(archive_path)
        return archive_path

    @staticmethod
    def get_or_extract(archive_path):
        """Given a path returned by add_local_model(), this method will return a tuple of
        (loaded_model, local_model_path).
        If this Python process ever loaded the model before, we will reuse that copy.
        """
        from pyspark.files import SparkFiles
        import zipfile

        if archive_path in _SparkDirectoryDistributor._extracted_dir_paths:
            return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]

        # BUG: Despite the documentation of SparkContext.addFile() and SparkFiles.get() in Scala
        # and Python, it turns out that we actually need to use the basename as the input to
        # SparkFiles.get(), as opposed to the (absolute) path.
        archive_path_basename = os.path.basename(archive_path)
        local_path = SparkFiles.get(archive_path_basename)
        temp_dir = tempfile.mkdtemp()
        zip_ref = zipfile.ZipFile(local_path, "r")
        zip_ref.extractall(temp_dir)
        zip_ref.close()

        _SparkDirectoryDistributor._extracted_dir_paths[archive_path] = temp_dir
        return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]
