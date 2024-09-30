import os
import subprocess


_global_dbconnect_artifact_cache = None


class DBConnectArtifactCache:
    """
    This class is designed for managing Databricks Connect artifacts cache.
    Note it doesn't support OSS Spark Connect.

    You can use this class in the following environment:
     - Databricks shared cluster python notebook REPL
     - Databricks Serverless python notebook REPL
     - Databricks connect client python REPL that connects to remote Databricks Serverless
     - Databricks connect client python REPL that connects to remote Databricks shared cluster

    .. code-block:: python
        :caption: Example

        # client side code
        db_artifact_cache = DBConnectArtifactCache.get_or_create()
        db_artifact_cache.add_artifact_archive("archive1", "/tmp/archive1.tar.gz")

        @pandas_udf(...)
        def my_udf(...):
           # we can get the unpacked archive files in `archive1_unpacked_dir`
           archive1_unpacked_dir = db_artifact_cache.get("archive1")
    """

    @classmethod
    def get_or_create(cls, spark):
        global _global_dbconnect_artifact_cache
        if _global_dbconnect_artifact_cache is None or spark is not _global_dbconnect_artifact_cache._spark:
            _global_dbconnect_artifact_cache = DBConnectArtifactCache(spark)
        return _global_dbconnect_artifact_cache

    def __init__(self, spark):
        self._spark = spark
        self._cache = {}

    def __getstate__(self):
        """
        The `DBConnectArtifactCache` instance is created in Databricks Connect client side,
        and it will be pickled to Databricks Connect UDF sandbox
        (see `get_unpacked_artifact_dir` method), but Spark Connect client object is
        not pickle-able, we need to skip this field.
        """
        state = self.__dict__.copy()
        # Don't pickle `_spark`
        del state["_spark"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._spark = None

    def has_cache_key(self, cache_key):
        return cache_key in self._cache

    def add_artifact_archive(self, cache_key, artifact_archive_path):
        """
        Add an artifact archive file to Databricks connect cache.
        The archive file must be 'tar.gz' format.
        You can only call this method in Databricks Connect client side.
        """
        if not artifact_archive_path.endswith(".tar.gz"):
            raise RuntimeError(
                "'add_artifact_archive' only supports archive file in 'tar.gz' format."
            )

        archive_file_name = os.path.basename(artifact_archive_path)
        if cache_key not in self._cache:
            self._spark.addArtfact(artifact_archive_path, archive=True)
            self._cache[cache_key] = archive_file_name

    def get_unpacked_artifact_dir(self, cache_key):
        """
        Get unpacked artifact directory path, you can only call this method
        inside Databricks Connect spark UDF sandbox.
        """
        if cache_key not in self._cache:
            raise RuntimeError(f"You haven't uploaded artifact '{cache_key}'.")
        archive_file_name = self._cache[cache_key]
        session_id = os.environ["DB_SESSION_UUID"]
        return f"/local_disk0/.ephemeral_nfs/artifacts/{session_id}/archives/{archive_file_name}"


def archive_directory(input_dir, archive_file_path):
    """
    Archive the `input_dir` directory, save the archive file to `archive_file_path`,
    the generated archive file is 'tar.gz' format.
    Note: all symlink files in the input directory are kept as it is in the archive file.
    """

    archive_file_path = os.path.abspath(archive_file_path)
    # Note: I don't use `shutil.make_archive` API because it replaces symlink files with
    #  the file symlink pointing to, which is not the expected behavior in our usage.
    #  We need to pack the python and virtualenv environment, which contains a bunch of
    #  symlink files.
    subprocess.check_call(
        f"tar -czf {archive_file_path} ./*",
        cwd=input_dir,
        shell=True
    )
    return archive_file_path


def extract_archive_to_dir(archive_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.check_call(
        ["tar", "-xf", archive_path, "-C", dest_dir]
    )
    return dest_dir

