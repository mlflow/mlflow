import atexit
import codecs
import errno
import fnmatch
import gzip
import importlib.util
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from types import TracebackType
from typing import Any
from urllib.parse import unquote
from urllib.request import pathname2url

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    _MLFLOW_MPD_NUM_RETRIES,
    _MLFLOW_MPD_RETRY_INTERVAL_SECONDS,
    MLFLOW_DOWNLOAD_CHUNK_TIMEOUT,
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import download_cloud_file_chunk
from mlflow.utils.databricks_utils import (
    get_databricks_local_temp_dir,
    get_databricks_nfs_temp_dir,
)
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status

ENCODING = "utf-8"
_PROGRESS_BAR_DISPLAY_THRESHOLD = 500_000_000  # 500 MB

_logger = logging.getLogger(__name__)

# This is for backward compatibility with databricks-feature-engineering<=0.10.2
if importlib.util.find_spec("yaml") is not None:
    try:
        from yaml import CSafeDumper as YamlSafeDumper
    except ImportError:
        from yaml import SafeDumper as YamlSafeDumper  # noqa: F401


class ArtifactProgressBar:
    def __init__(self, desc, total, step, **kwargs) -> None:
        self.desc = desc
        self.total = total
        self.step = step
        self.pbar = None
        self.progress = 0
        self.kwargs = kwargs

    def set_pbar(self):
        if MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR.get():
            try:
                from tqdm.auto import tqdm

                self.pbar = tqdm(total=self.total, desc=self.desc, **self.kwargs)
            except ImportError:
                pass

    @classmethod
    def chunks(cls, file_size, desc, chunk_size):
        bar = cls(
            desc,
            total=file_size,
            step=chunk_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
        )
        if file_size >= _PROGRESS_BAR_DISPLAY_THRESHOLD:
            bar.set_pbar()
        return bar

    @classmethod
    def files(cls, desc, total):
        bar = cls(desc, total=total, step=1)
        bar.set_pbar()
        return bar

    def update(self):
        if self.pbar:
            update_step = min(self.total - self.progress, self.step)
            self.pbar.update(update_step)
            self.pbar.refresh()
            self.progress += update_step

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()


def is_directory(name):
    return os.path.isdir(name)


def is_file(name):
    return os.path.isfile(name)


def exists(name):
    return os.path.exists(name)


def list_all(root, filter_func=lambda x: True, full_path=False):
    """List all entities directly under 'dir_name' that satisfy 'filter_func'

    Args:
        root: Name of directory to start search.
        filter_func: function or lambda that takes path.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all files or directories that satisfy the criteria.

    """
    if not is_directory(root):
        raise Exception(f"Invalid parent directory '{root}'")
    matches = [x for x in os.listdir(root) if filter_func(os.path.join(root, x))]
    return [os.path.join(root, m) for m in matches] if full_path else matches


def list_subdirs(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type d``

    Args:
        dir_name: Name of directory to start search.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all directories directly under 'dir_name'.
    """
    return list_all(dir_name, os.path.isdir, full_path)


def list_files(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type f``

    Args:
        dir_name: Name of directory to start search.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all files directly under 'dir_name'.
    """
    return list_all(dir_name, os.path.isfile, full_path)


def find(root, name, full_path=False):
    """Search for a file in a root directory. Equivalent to:
      ``find $root -name "$name" -depth 1``

    Args:
        root: Name of root directory for find.
        name: Name of file or directory to find directly under root directory.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of matching files or directories.
    """
    path_name = os.path.join(root, name)
    return list_all(root, lambda x: x == path_name, full_path)


def mkdir(root, name=None):
    """Make directory with name "root/name", or just "root" if name is None.

    Args:
        root: Name of parent directory.
        name: Optional name of leaf directory.

    Returns:
        Path to created directory.
    """
    target = os.path.join(root, name) if name is not None else root
    try:
        os.makedirs(target, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(target):
            raise e
    return target


def make_containing_dirs(path):
    """
    Create the base directory for a given file path if it does not exist; also creates parent
    directories.
    """
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def read_parquet_as_pandas_df(data_parquet_path: str):
    """Deserialize and load the specified parquet file as a Pandas DataFrame.

    Args:
        data_parquet_path: String, path object (implementing os.PathLike[str]),
            or file-like object implementing a binary read() function. The string
            could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
            For file URLs, a host is expected. A local file could
            be: file://localhost/path/to/table.parquet. A file URL can also be a path to a
            directory that contains multiple partitioned parquet files. Pyarrow
            support paths to directories as well as file URLs. A directory
            path could be: file://localhost/path/to/tables or s3://bucket/partition_dir.

    Returns:
        pandas dataframe
    """
    import pandas as pd

    return pd.read_parquet(data_parquet_path, engine="pyarrow")


def write_pandas_df_as_parquet(df, data_parquet_path: str):
    """Write a DataFrame to the binary parquet format.

    Args:
        df: pandas data frame.
        data_parquet_path: String, path object (implementing os.PathLike[str]),
            or file-like object implementing a binary write() function.

    """
    df.to_parquet(data_parquet_path, engine="pyarrow")


class TempDir:
    def __init__(self, chdr=False, remove_on_exit=True):
        self._dir = None
        self._path = None
        self._chdr = chdr
        self._remove = remove_on_exit

    def __enter__(self):
        self._path = os.path.abspath(create_tmp_dir())
        assert os.path.exists(self._path)
        if self._chdr:
            self._dir = os.path.abspath(os.getcwd())
            os.chdir(self._path)
        return self

    def __exit__(self, tp, val, traceback):
        if self._chdr and self._dir:
            os.chdir(self._dir)
            self._dir = None
        if self._remove and os.path.exists(self._path):
            shutil.rmtree(self._path)

        assert not self._remove or not os.path.exists(self._path)

    def path(self, *path):
        return os.path.join("./", *path) if self._chdr else os.path.join(self._path, *path)


def read_file_lines(parent_path, file_name):
    """Return the contents of the file as an array where each element is a separate line.

    Args:
        parent_path: Full path to the directory that contains the file.
        file_name: Leaf file name.

    Returns:
        All lines in the file as an array.

    """
    file_path = os.path.join(parent_path, file_name)
    with codecs.open(file_path, mode="r", encoding=ENCODING) as f:
        return f.readlines()


def read_file(parent_path, file_name):
    """Return the contents of the file.

    Args:
        parent_path: Full path to the directory that contains the file.
        file_name: Leaf file name.

    Returns:
        The contents of the file.

    """
    file_path = os.path.join(parent_path, file_name)
    with codecs.open(file_path, mode="r", encoding=ENCODING) as f:
        return f.read()


def get_file_info(path, rel_path):
    """Returns file meta data : location, size, ... etc

    Args:
        path: Path to artifact.
        rel_path: Relative path.

    Returns:
        `FileInfo` object
    """
    if is_directory(path):
        return FileInfo(rel_path, True, None)
    else:
        return FileInfo(rel_path, False, os.path.getsize(path))


def get_relative_path(root_path, target_path):
    """Remove root path common prefix and return part of `path` relative to `root_path`.

    Args:
        root_path: Root path.
        target_path: Desired path for common prefix removal.

    Returns:
        Path relative to root_path.
    """
    if len(root_path) > len(target_path):
        raise Exception(f"Root path '{root_path}' longer than target path '{target_path}'")
    common_prefix = os.path.commonprefix([root_path, target_path])
    return os.path.relpath(target_path, common_prefix)


def mv(target, new_parent):
    shutil.move(target, new_parent)


def write_to(filename, data):
    with codecs.open(filename, mode="w", encoding=ENCODING) as handle:
        handle.write(data)


def append_to(filename, data):
    with open(filename, "a") as handle:
        handle.write(data)


def make_tarfile(output_filename, source_dir, archive_name, custom_filter=None):
    # Helper for filtering out modification timestamps
    def _filter_timestamps(tar_info):
        tar_info.mtime = 0
        return tar_info if custom_filter is None else custom_filter(tar_info)

    unzipped_file_handle, unzipped_filename = tempfile.mkstemp()
    try:
        with tarfile.open(unzipped_filename, "w") as tar:
            tar.add(source_dir, arcname=archive_name, filter=_filter_timestamps)
        # When gzipping the tar, don't include the tar's filename or modification time in the
        # zipped archive (see https://docs.python.org/3/library/gzip.html#gzip.GzipFile)
        with (
            gzip.GzipFile(
                filename="", fileobj=open(output_filename, "wb"), mode="wb", mtime=0
            ) as gzipped_tar,
            open(unzipped_filename, "rb") as tar,
        ):
            gzipped_tar.write(tar.read())
    finally:
        os.close(unzipped_file_handle)


def _copy_project(src_path, dst_path=""):
    """Internal function used to copy MLflow project during development.

    Copies the content of the whole directory tree except patterns defined in .dockerignore.
    The MLflow is assumed to be accessible as a local directory in this case.

    Args:
        src_path: Path to the original MLflow project
        dst_path: MLflow will be copied here

    Returns:
        Name of the MLflow project directory.
    """

    def _docker_ignore(mlflow_root):
        docker_ignore = os.path.join(mlflow_root, ".dockerignore")
        patterns = []
        if os.path.exists(docker_ignore):
            with open(docker_ignore) as f:
                patterns = [x.strip() for x in f]

        def ignore(_, names):
            res = set()
            for p in patterns:
                res.update(set(fnmatch.filter(names, p)))
            return list(res)

        return ignore if patterns else None

    mlflow_dir = "mlflow-project"
    # check if we have project root
    assert os.path.isfile(os.path.join(src_path, "pyproject.toml")), "file not found " + str(
        os.path.abspath(os.path.join(src_path, "pyproject.toml"))
    )
    shutil.copytree(src_path, os.path.join(dst_path, mlflow_dir), ignore=_docker_ignore(src_path))
    return mlflow_dir


def _copy_file_or_tree(src, dst, dst_dir=None):
    """
    Returns:
        The path to the copied artifacts, relative to `dst`.
    """
    dst_subpath = os.path.basename(os.path.abspath(src))
    if dst_dir is not None:
        dst_subpath = os.path.join(dst_dir, dst_subpath)
    dst_path = os.path.join(dst, dst_subpath)
    if os.path.isfile(src):
        dst_dirpath = os.path.dirname(dst_path)
        if not os.path.exists(dst_dirpath):
            os.makedirs(dst_dirpath)
        shutil.copy(src=src, dst=dst_path)
    else:
        shutil.copytree(src=src, dst=dst_path, ignore=shutil.ignore_patterns("__pycache__"))
    return dst_subpath


def _get_local_project_dir_size(project_path):
    """Internal function for reporting the size of a local project directory before copying to
    destination for cli logging reporting to stdout.

    Args:
        project_path: local path of the project directory

    Returns:
        directory file sizes in KB, rounded to single decimal point for legibility
    """

    total_size = 0
    for root, _, files in os.walk(project_path):
        for f in files:
            path = os.path.join(root, f)
            total_size += os.path.getsize(path)
    return round(total_size / 1024.0, 1)


def _get_local_file_size(file):
    """
    Get the size of a local file in KB
    """
    return round(os.path.getsize(file) / 1024.0, 1)


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def relative_path_to_artifact_path(path):
    if os.path == posixpath:
        return path
    if os.path.abspath(path) == path:
        raise Exception("This method only works with relative paths.")
    return unquote(pathname2url(path))


def path_to_local_file_uri(path):
    """
    Convert local filesystem path to local file uri.
    """
    return pathlib.Path(os.path.abspath(path)).as_uri()


def path_to_local_sqlite_uri(path):
    """
    Convert local filesystem path to sqlite uri.
    """
    path = posixpath.abspath(pathname2url(os.path.abspath(path)))
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:///"
    return prefix + path


def local_file_uri_to_path(uri):
    """
    Convert URI to local filesystem path.
    No-op if the uri does not have the expected scheme.
    """
    path = uri
    if uri.startswith("file:"):
        parsed_path = urllib.parse.urlparse(uri)
        path = parsed_path.path
        # Fix for retaining server name in UNC path.
        if is_windows() and parsed_path.netloc:
            return urllib.request.url2pathname(rf"\\{parsed_path.netloc}{path}")
    return urllib.request.url2pathname(path)


def get_local_path_or_none(path_or_uri):
    """Check if the argument is a local path (no scheme or file:///) and return local path if true,
    None otherwise.
    """
    parsed_uri = urllib.parse.urlparse(path_or_uri)
    if len(parsed_uri.scheme) == 0 or parsed_uri.scheme == "file" and len(parsed_uri.netloc) == 0:
        return local_file_uri_to_path(path_or_uri)
    else:
        return None


def yield_file_in_chunks(file, chunk_size=100000000):
    """
    Generator to chunk-ify the inputted file based on the chunk-size.
    """
    with open(file, "rb") as f:
        while True:
            if chunk := f.read(chunk_size):
                yield chunk
            else:
                break


def download_file_using_http_uri(http_uri, download_path, chunk_size=100000000, headers=None):
    """
    Downloads a file specified using the `http_uri` to a local `download_path`. This function
    uses a `chunk_size` to ensure an OOM error is not raised a large file is downloaded.

    Note : This function is meant to download files using presigned urls from various cloud
            providers.
    """
    if headers is None:
        headers = {}
    with cloud_storage_http_request("get", http_uri, stream=True, headers=headers) as response:
        augmented_raise_for_status(response)
        with open(download_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    break
                output_file.write(chunk)


@dataclass(frozen=True)
class _Chunk:
    index: int
    start: int
    end: int
    path: str


def _yield_chunks(path, file_size, chunk_size):
    num_requests = int(math.ceil(file_size / float(chunk_size)))
    for i in range(num_requests):
        range_start = i * chunk_size
        range_end = min(range_start + chunk_size - 1, file_size - 1)
        yield _Chunk(i, range_start, range_end, path)


def parallelized_download_file_using_http_uri(
    thread_pool_executor,
    http_uri,
    download_path,
    remote_file_path,
    file_size,
    uri_type,
    chunk_size,
    env,
    headers=None,
):
    """
    Downloads a file specified using the `http_uri` to a local `download_path`. This function
    sends multiple requests in parallel each specifying its own desired byte range as a header,
    then reconstructs the file from the downloaded chunks. This allows for downloads of large files
    without OOM risk.

    Note : This function is meant to download files using presigned urls from various cloud
            providers.
    Returns a dict of chunk index : exception, if one was thrown for that index.
    """

    def run_download(chunk: _Chunk):
        try:
            subprocess.run(
                [
                    sys.executable,
                    download_cloud_file_chunk.__file__,
                    "--range-start",
                    str(chunk.start),
                    "--range-end",
                    str(chunk.end),
                    "--headers",
                    json.dumps(headers or {}),
                    "--download-path",
                    download_path,
                    "--http-uri",
                    http_uri,
                ],
                text=True,
                check=True,
                capture_output=True,
                timeout=MLFLOW_DOWNLOAD_CHUNK_TIMEOUT.get(),
                env=env,
            )
        except (TimeoutExpired, CalledProcessError) as e:
            raise MlflowException(
                f"""
----- stdout -----
{e.stdout.strip()}

----- stderr -----
{e.stderr.strip()}
"""
            ) from e

    chunks = _yield_chunks(remote_file_path, file_size, chunk_size)
    # Create file if it doesn't exist or erase the contents if it does. We should do this here
    # before sending to the workers so they can each individually seek to their respective positions
    # and write chunks without overwriting.
    with open(download_path, "w"):
        pass
    if uri_type == ArtifactCredentialType.GCP_SIGNED_URL or uri_type is None:
        chunk = next(chunks)
        # GCP files could be transcoded, in which case the range header is ignored.
        # Test if this is the case by downloading one chunk and seeing if it's larger than the
        # requested size. If yes, let that be the file; if not, continue downloading more chunks.
        download_chunk(
            range_start=chunk.start,
            range_end=chunk.end,
            headers=headers,
            download_path=download_path,
            http_uri=http_uri,
        )
        downloaded_size = os.path.getsize(download_path)
        # If downloaded size was equal to the chunk size it would have been downloaded serially,
        # so we don't need to consider this here
        if downloaded_size > chunk_size:
            return {}

    futures = {thread_pool_executor.submit(run_download, chunk): chunk for chunk in chunks}
    failed_downloads = {}
    with ArtifactProgressBar.chunks(file_size, f"Downloading {download_path}", chunk_size) as pbar:
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                future.result()
            except Exception as e:
                _logger.debug(
                    f"Failed to download chunk {chunk.index} for {chunk.path}: {e}. "
                    f"The download of this chunk will be retried later."
                )
                failed_downloads[chunk] = future.exception()
            else:
                pbar.update()

    return failed_downloads


def download_chunk_retries(*, chunks, http_uri, headers, download_path):
    num_retries = _MLFLOW_MPD_NUM_RETRIES.get()
    interval = _MLFLOW_MPD_RETRY_INTERVAL_SECONDS.get()
    for chunk in chunks:
        _logger.info(f"Retrying download of chunk {chunk.index} for {chunk.path}")
        for retry in range(num_retries):
            try:
                download_chunk(
                    range_start=chunk.start,
                    range_end=chunk.end,
                    headers=headers,
                    download_path=download_path,
                    http_uri=http_uri,
                )
                _logger.info(f"Successfully downloaded chunk {chunk.index} for {chunk.path}")
                break
            except Exception:
                if retry == num_retries - 1:
                    raise
            time.sleep(interval)


def _handle_readonly_on_windows(func, path, exc_info):
    """
    This function should not be called directly but should be passed to `onerror` of
    `shutil.rmtree` in order to reattempt the removal of a read-only file after making
    it writable on Windows.

    References:
    - https://bugs.python.org/issue19643
    - https://bugs.python.org/issue43657
    """
    exc_type, exc_value = exc_info[:2]
    should_reattempt = (
        is_windows()
        and func in (os.unlink, os.rmdir)
        and issubclass(exc_type, PermissionError)
        and exc_value.winerror == 5
    )
    if not should_reattempt:
        raise exc_value
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _get_tmp_dir():
    from mlflow.utils.databricks_utils import get_repl_id, is_in_databricks_runtime

    if is_in_databricks_runtime():
        try:
            return get_databricks_local_temp_dir()
        except Exception:
            pass

        if repl_id := get_repl_id():
            return os.path.join("/tmp", "repl_tmp_data", repl_id)

    return None


def create_tmp_dir():
    if directory := _get_tmp_dir():
        os.makedirs(directory, exist_ok=True)
        return tempfile.mkdtemp(dir=directory)

    return tempfile.mkdtemp()


@cache_return_value_per_process
def get_or_create_tmp_dir():
    """
    Get or create a temporary directory which will be removed once python process exit.
    """
    from mlflow.utils.databricks_utils import get_repl_id, is_in_databricks_runtime

    if is_in_databricks_runtime() and get_repl_id() is not None:
        # Note: For python process attached to databricks notebook, atexit does not work.
        # The directory returned by `get_databricks_local_tmp_dir`
        # will be removed once databricks notebook detaches.
        # The temp directory is designed to be used by all kinds of applications,
        # so create a child directory "mlflow" for storing mlflow temp data.
        try:
            repl_local_tmp_dir = get_databricks_local_temp_dir()
        except Exception:
            repl_local_tmp_dir = os.path.join("/tmp", "repl_tmp_data", get_repl_id())

        tmp_dir = os.path.join(repl_local_tmp_dir, "mlflow")
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp()
        # mkdtemp creates a directory with permission 0o700
        # For Spark UDFs, we need to make it accessible to other processes
        # Use 0o750 (owner: rwx, group: r-x, others: None) instead of 0o777
        # This allows read/execute but not write for group and others
        os.chmod(tmp_dir, 0o750)
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)

    return tmp_dir


@cache_return_value_per_process
def get_or_create_nfs_tmp_dir():
    """
    Get or create a temporary NFS directory which will be removed once python process exit.
    """
    from mlflow.utils.databricks_utils import get_repl_id, is_in_databricks_runtime
    from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir

    nfs_root_dir = get_nfs_cache_root_dir()

    if is_in_databricks_runtime() and get_repl_id() is not None:
        # Note: In databricks, atexit hook does not work.
        # The directory returned by `get_databricks_nfs_tmp_dir`
        # will be removed once databricks notebook detaches.
        # The temp directory is designed to be used by all kinds of applications,
        # so create a child directory "mlflow" for storing mlflow temp data.
        try:
            repl_nfs_tmp_dir = get_databricks_nfs_temp_dir()
        except Exception:
            repl_nfs_tmp_dir = os.path.join(nfs_root_dir, "repl_tmp_data", get_repl_id())

        tmp_nfs_dir = os.path.join(repl_nfs_tmp_dir, "mlflow")
        os.makedirs(tmp_nfs_dir, exist_ok=True)
    else:
        tmp_nfs_dir = tempfile.mkdtemp(dir=nfs_root_dir)
        # mkdtemp creates a directory with permission 0o700
        # change it to be 0o777 to ensure it can be seen in spark UDF
        os.chmod(tmp_nfs_dir, 0o777)
        atexit.register(shutil.rmtree, tmp_nfs_dir, ignore_errors=True)

    return tmp_nfs_dir


def write_spark_dataframe_to_parquet_on_local_disk(spark_df, output_path):
    """Write spark dataframe in parquet format to local disk.

    Args:
        spark_df: Spark dataframe.
        output_path: Path to write the data to.

    """
    from mlflow.utils.databricks_utils import is_in_databricks_runtime

    if is_in_databricks_runtime():
        dbfs_path = os.path.join(".mlflow", "cache", str(uuid.uuid4()))
        spark_df.coalesce(1).write.format("parquet").save(dbfs_path)
        shutil.copytree("/dbfs/" + dbfs_path, output_path)
        shutil.rmtree("/dbfs/" + dbfs_path)
    else:
        spark_df.coalesce(1).write.format("parquet").save(output_path)


def shutil_copytree_without_file_permissions(src_dir, dst_dir):
    """
    Copies the directory src_dir into dst_dir, without preserving filesystem permissions
    """
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for dirname in dirnames:
            relative_dir_path = os.path.relpath(os.path.join(dirpath, dirname), src_dir)
            # For each directory <dirname> immediately under <dirpath>, create an equivalently-named
            # directory under the destination directory
            abs_dir_path = os.path.join(dst_dir, relative_dir_path)
            if not os.path.exists(abs_dir_path):
                os.mkdir(abs_dir_path)
        for filename in filenames:
            # For each file with name <filename> immediately under <dirpath>, copy that file to
            # the appropriate location in the destination directory
            file_path = os.path.join(dirpath, filename)
            relative_file_path = os.path.relpath(file_path, src_dir)
            abs_file_path = os.path.join(dst_dir, relative_file_path)
            shutil.copy2(file_path, abs_file_path)


def contains_path_separator(path):
    """
    Returns True if a path contains a path separator, False otherwise.
    """
    return any((sep in path) for sep in (os.path.sep, os.path.altsep) if sep is not None)


def contains_percent(path):
    """
    Returns True if a path contains a percent character, False otherwise.
    """
    return "%" in path


def read_chunk(path: os.PathLike, size: int, start_byte: int = 0) -> bytes:
    """Read a chunk of bytes from a file.

    Args:
        path: Path to the file.
        size: The size of the chunk.
        start_byte: The start byte of the chunk.

    Returns:
        The chunk of bytes.

    """
    with open(path, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
        return f.read(size)


@contextmanager
def remove_on_error(path: os.PathLike, onerror=None):
    """A context manager that removes a file or directory if an exception is raised during
    execution.

    Args:
        path: Path to the file or directory.
        onerror: A callback function that will be called with the captured exception before
            the file or directory is removed. For example, you can use this callback to
            log the exception.

    """
    try:
        yield
    except Exception as e:
        if onerror:
            onerror(e)
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        _logger.warning(
            f"Failed to remove {path}" if os.path.exists(path) else f"Successfully removed {path}"
        )
        raise


@contextmanager
def chdir(path: str) -> None:
    """Temporarily change the current working directory to the specified path.

    Args:
        path: The path to use as the temporary working directory.
    """
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def get_total_file_size(path: str | pathlib.Path) -> int | None:
    """Return the size of all files under given path, including files in subdirectories.

    Args:
        path: The absolute path of a local directory.

    Returns:
        size in bytes.

    """
    try:
        if isinstance(path, pathlib.Path):
            path = str(path)
        if not os.path.exists(path):
            raise MlflowException(
                message=f"The given {path} does not exist.", error_code=INVALID_PARAMETER_VALUE
            )
        if not os.path.isdir(path):
            raise MlflowException(
                message=f"The given {path} is not a directory.", error_code=INVALID_PARAMETER_VALUE
            )

        total_size = 0
        for cur_path, dirs, files in os.walk(path):
            full_paths = [os.path.join(cur_path, file) for file in files]
            total_size += sum(map(os.path.getsize, full_paths))
        return total_size
    except Exception as e:
        _logger.info(f"Failed to get the total size of {path} because of error :{e}")
        return None


def write_yaml(
    root: str,
    file_name: str,
    data: dict[str, Any],
    overwrite: bool = False,
    sort_keys: bool = True,
    ensure_yaml_extension: bool = True,
) -> None:
    """
    NEVER TOUCH THIS FUNCTION. KEPT FOR BACKWARD COMPATIBILITY with
    databricks-feature-engineering<=0.10.2
    """
    import yaml

    with open(os.path.join(root, file_name), "w") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys,
        )


def read_yaml(root: str, file_name: str) -> dict[str, Any]:
    """
    NEVER TOUCH THIS FUNCTION. KEPT FOR BACKWARD COMPATIBILITY with
    databricks-feature-engineering<=0.10.2
    """
    import yaml

    with open(os.path.join(root, file_name)) as f:
        return yaml.safe_load(f)


class ExclusiveFileLock:
    """
    Exclusive file lock (only works on Unix system)
    """

    def __init__(self, path: str):
        if os.name == "nt":
            raise MlflowException("ExclusiveFileLock class does not support Windows system.")
        self.path = path
        self.fd = None

    def __enter__(self) -> None:
        # Python on Windows does not have `fcntl` module, so importing it lazily.
        import fcntl  # clint: disable=lazy-builtin-import

        # Open file (create if missing)
        self.fd = open(self.path, "w")
        # Acquire exclusive lock (blocking)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        # Python on Windows does not have `fcntl` module, so importing it lazily.
        import fcntl  # clint: disable=lazy-builtin-import

        # Release lock
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()
