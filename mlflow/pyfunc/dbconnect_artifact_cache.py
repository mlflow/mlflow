import json
import os
import shutil
import tarfile
from contextlib import ExitStack
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import check_tarfile_security, get_or_create_tmp_dir, remove_on_error

_CACHE_MAP_FILE_NAME = "db_connect_artifact_cache.json"


class DBConnectArtifactCache:
    """
    Manages Databricks Connect artifacts cache.
    Note it doesn't support OSS Spark Connect.

    This class can be used in the following environment:
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
        def my_udf(x):
            # we can get the unpacked archive files in `archive1_unpacked_dir`
            archive1_unpacked_dir = db_artifact_cache.get("archive1")
    """

    _global_cache = None

    @staticmethod
    def get_or_create(spark):
        if (
            DBConnectArtifactCache._global_cache is None
            or spark is not DBConnectArtifactCache._global_cache._spark
        ):
            DBConnectArtifactCache._global_cache = DBConnectArtifactCache(spark)
            cache_file = os.path.join(get_or_create_tmp_dir(), _CACHE_MAP_FILE_NAME)
            if is_in_databricks_runtime() and os.path.exists(cache_file):
                # In databricks runtime (shared cluster or Serverless), when you restart the
                # notebook REPL by %restart_python or dbutils.library.restartPython(), the
                # DBConnect session is still preserved. So in this case, we can reuse the cached
                # artifact files.
                # So that when adding artifact, the cache map is serialized to local disk file
                # `db_connect_artifact_cache.json` and after REPL restarts,
                # `DBConnectArtifactCache` restores the cache map by loading data from the file.
                with open(cache_file) as f:
                    DBConnectArtifactCache._global_cache._cache = json.load(f)
        return DBConnectArtifactCache._global_cache

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
            self._spark.addArtifact(artifact_archive_path, archive=True)
            self._cache[cache_key] = archive_file_name

        if is_in_databricks_runtime():
            with open(os.path.join(get_or_create_tmp_dir(), _CACHE_MAP_FILE_NAME), "w") as f:
                json.dump(self._cache, f)

    def get_unpacked_artifact_dir(self, cache_key):
        """
        Get unpacked artifact directory path, you can only call this method
        inside Databricks Connect spark UDF sandbox.
        """
        from pyspark.taskcontext import TaskContext

        if cache_key not in self._cache:
            raise RuntimeError(f"The artifact '{cache_key}' does not exist.")
        archive_file_name = self._cache[cache_key]

        if session_id := os.environ.get("DB_SESSION_UUID"):
            task_context = TaskContext.get()
            if hasattr(task_context, "artifactDir"):
                return os.path.join(task_context.artifactDir(), "archives", archive_file_name)

            return (
                f"/local_disk0/.ephemeral_nfs/artifacts/{session_id}/archives/{archive_file_name}"
            )

        # If 'DB_SESSION_UUID' environment variable does not exist, it means it is running
        # in a dedicated mode Spark cluster.
        return os.path.join(os.getcwd(), archive_file_name)


def archive_directory(input_dir, archive_file_path):
    """
    Archive the `input_dir` directory, save the archive file to `archive_file_path`,
    the generated archive file is 'tar.gz' format.
    Note: all symlink files in the input directory are kept as it is in the archive file.
    """

    archive_file_path = os.path.abspath(archive_file_path)
    # Note: `shutil.make_archive` doesn't work because it replaces symlink files with
    #  the file symlink pointing to, which is not the expected behavior in our usage.
    #  We need to pack the python and virtualenv environment, which contains a bunch of
    #  symlink files.
    # Use `tarfile` instead of the `tar` CLI so entry names are always treated literally:
    #  GNU tar parses names starting with "-" as options and decodes backslash escapes
    #  (e.g. "\056\056\057x" becomes "../x"), which can archive files outside `input_dir`.
    with tarfile.open(archive_file_path, "w:gz", compresslevel=6, dereference=False) as tar:
        for name in os.listdir(input_dir):
            tar.add(os.path.join(input_dir, name), arcname=name)
    return archive_file_path


def extract_archive_to_dir(archive_path, dest_dir):
    check_tarfile_security(archive_path)
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        _safe_extractall(tar, dest_dir)
    return dest_dir


def _safe_extractall(tar, dest_dir):
    resolved_dest = Path(dest_dir).resolve()

    def is_regular_file_within_dest(path: Path) -> bool:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(resolved_dest)
        )

    for member in tar.getmembers():
        member_path = (resolved_dest / member.name).resolve()
        # A hard link is created on the file its target resolves to, following any symlink
        # already extracted, so the resolved target must stay inside the destination too.
        # When the target is missing (e.g. a dangling symlink), tarfile instead re-extracts
        # the target member at the link's path, so require an existing regular file.
        if not member_path.is_relative_to(resolved_dest) or (
            member.islnk() and not is_regular_file_within_dest(resolved_dest / member.linkname)
        ):
            raise MlflowException.invalid_parameter_value(
                f"Tar archive member {member.name!r} would be extracted outside "
                f"the destination directory."
            )
        if member.islnk():
            try:
                _extract_hardlink(member, resolved_dest)
            except OSError as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to extract hard link {member.name!r} -> {member.linkname!r}: {e}"
                ) from e
        else:
            tar.extract(member, path=resolved_dest)


def _extract_hardlink(member, resolved_dest):
    # Keep the original path spelling so that os.link and open resolve the same target.
    target_path = os.path.join(resolved_dest, member.linkname)
    link_path = os.path.join(resolved_dest, member.name)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    try:
        os.link(target_path, link_path)
    except OSError:
        # Filesystems report unsupported hard links with different errors. Copy the
        # validated target to a new file without re-extracting an archive member.
        with ExitStack() as cleanup:
            with open(target_path, "rb") as source, open(link_path, "xb") as destination:
                cleanup.enter_context(remove_on_error(link_path))
                shutil.copyfileobj(source, destination)
            shutil.copystat(target_path, link_path)
