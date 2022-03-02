import os.path
import tempfile
import uuid
import atexit
import shutil
from pyspark.files import SparkFiles


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


def _spark_context_add_archive(spark, file_path):
    spark.sparkContext._jsc.sc().addArchive(file_path)


class _SparkBroadcastFileCache:
    """
    This class provide helper method for broadcasting driver side file/directory to
    any spark executors.
    The file/directory is archived before broadcasting to executor side.

    Note:
        For databricks runtime, don't use this class but use NFS instead.
    """

    _file_path_to_cache_key_map = {}

    _archive_tmp_dir = None

    @staticmethod
    def _get_or_create_archive_tmp_dir():
        if _SparkBroadcastFileCache._archive_tmp_dir is None:
            _SparkBroadcastFileCache._archive_tmp_dir = \
                _SparkBroadcastFileCache._archive_tmp_dir = tempfile.mkdtemp()

            # TODO:
            #  in atexit handler, also remove these cache files in spark executor side.
            #  this require spark add corresponding removal API first.
            #  otherwise these cache file will exist until spark application shutdown.
            #  Note in databricks runtime spark application keep running until
            #  spark cluster shutdown.
            atexit.register(
                shutil.rmtree, _SparkBroadcastFileCache._archive_tmp_dir, ignore_errors=True
            )
        return _SparkBroadcastFileCache._archive_tmp_dir

    @staticmethod
    def add_file(file_path):
        """
        This method is only allowed called from spark application driver side.
        The file path can be a normal file or a directory path.
        return cache_key as a string.
        """
        if file_path in _SparkBroadcastFileCache._file_path_to_cache_key_map:
            return _SparkBroadcastFileCache._file_path_to_cache_key_map[file_path]
        file_path = os.path.normpath(file_path)
        archive_dir = _SparkBroadcastFileCache._get_or_create_archive_tmp_dir()
        archive_basename = uuid.uuid4().hex
        # NB: We must archive the directory as `Spark.addFile` does not support non-DFS
        # directories when recursive=True.
        archive_path = shutil.make_archive(os.path.join(archive_dir, archive_basename), "zip", file_path)
        # Use `sparkContext.addArchive` API instead of `sparkContext.addFile` API because:
        #  addArchive will auto extract archive and return unarchived path when get file from executor side
        #  as opposed, using `sparkContext.addFile` API we need write extracting archive logic code by
        #  ourselves, this causes several issues:
        #   1. If we cache the extracted files within python process scope,
        #   If `spark.python.worker.reuse` is False (the case databricks MLR by default),
        #   then every spark task will extract the file again and save it to a new temporary directory,
        #   it wastes both cpu and disk a lot.
        #   2. If we cache the extracted files globally and allow it being shared with multiple
        #   python processes (udf tasks), then we are hard to handle race condition. Each udf task run
        #   within an individual python process and they are hard to coordinate with all others.
        _spark_context_add_archive(_get_active_spark_session(), archive_path)
        # Note: the cache key is in the format "{uuid}.zip", we must ensure the cache key is unique
        # for each file_path
        cache_key = os.path.basename(archive_path)
        _SparkBroadcastFileCache._file_path_to_cache_key_map[file_path] = cache_key
        return cache_key

    @staticmethod
    def get_file(cache_key):
        """
        This method can be called from spark task routine.
        The file_path must be the same value with the path you called `add_file` from driver side.
        Return the unarchived file/directory.
        """
        return SparkFiles.get(cache_key)
