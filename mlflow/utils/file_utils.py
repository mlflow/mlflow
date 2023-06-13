import codecs
import errno
import gzip
import json
import math
import os
import posixpath
import shutil
import sys
import tarfile
import tempfile
import stat
import pathlib
from concurrent.futures import as_completed
from contextlib import contextmanager
import uuid
import fnmatch

import urllib.parse
import urllib.request
from urllib.parse import unquote
from urllib.request import pathname2url


import atexit

import yaml

try:
    from yaml import CSafeLoader as YamlSafeLoader, CSafeDumper as YamlSafeDumper
except ImportError:
    from yaml import SafeLoader as YamlSafeLoader, SafeDumper as YamlSafeDumper

from mlflow.entities import FileInfo
from mlflow.exceptions import MissingConfigException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.process import cache_return_value_per_process, _exec_cmd
from mlflow.utils import merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils import download_cloud_file_chunk
from mlflow.utils.request_utils import download_chunk


ENCODING = "utf-8"
MAX_PARALLEL_DOWNLOAD_WORKERS = os.cpu_count() * 2


def is_directory(name):
    return os.path.isdir(name)


def is_file(name):
    return os.path.isfile(name)


def exists(name):
    return os.path.exists(name)


def list_all(root, filter_func=lambda x: True, full_path=False):
    """
    List all entities directly under 'dir_name' that satisfy 'filter_func'

    :param root: Name of directory to start search
    :param filter_func: function or lambda that takes path
    :param full_path: If True will return results as full path including `root`

    :return: list of all files or directories that satisfy the criteria.
    """
    if not is_directory(root):
        raise Exception("Invalid parent directory '%s'" % root)
    matches = [x for x in os.listdir(root) if filter_func(os.path.join(root, x))]
    return [os.path.join(root, m) for m in matches] if full_path else matches


def list_subdirs(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type d``

    :param dir_name: Name of directory to start search
    :param full_path: If True will return results as full path including `root`

    :return: list of all directories directly under 'dir_name'
    """
    return list_all(dir_name, os.path.isdir, full_path)


def list_files(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type f``

    :param dir_name: Name of directory to start search
    :param full_path: If True will return results as full path including `root`

    :return: list of all files directly under 'dir_name'
    """
    return list_all(dir_name, os.path.isfile, full_path)


def find(root, name, full_path=False):
    """
    Search for a file in a root directory. Equivalent to:
      ``find $root -name "$name" -depth 1``

    :param root: Name of root directory for find
    :param name: Name of file or directory to find directly under root directory
    :param full_path: If True will return results as full path including `root`

    :return: list of matching files or directories
    """
    path_name = os.path.join(root, name)
    return list_all(root, lambda x: x == path_name, full_path)


def mkdir(root, name=None):
    """
    Make directory with name "root/name", or just "root" if name is None.

    :param root: Name of parent directory
    :param name: Optional name of leaf directory

    :return: Path to created directory
    """
    target = os.path.join(root, name) if name is not None else root
    try:
        os.makedirs(target)
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


def write_yaml(root, file_name, data, overwrite=False, sort_keys=True):
    """
    Write dictionary data in yaml format.

    :param root: Directory name.
    :param file_name: Desired file name. Will automatically add .yaml extension if not given
    :param data: data to be dumped as yaml format
    :param overwrite: If True, will overwrite existing files
    """
    if not exists(root):
        raise MissingConfigException("Parent directory '%s' does not exist." % root)

    file_path = os.path.join(root, file_name)
    yaml_file_name = file_path if file_path.endswith(".yaml") else file_path + ".yaml"

    if exists(yaml_file_name) and not overwrite:
        raise Exception(f"Yaml file '{file_path}' exists as '{yaml_file_name}")

    try:
        with codecs.open(yaml_file_name, mode="w", encoding=ENCODING) as yaml_file:
            yaml.dump(
                data,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=sort_keys,
                Dumper=YamlSafeDumper,
            )
    except Exception as e:
        raise e


def overwrite_yaml(root, file_name, data):
    """
    Safely overwrites a preexisting yaml file, ensuring that file contents are not deleted or
    corrupted if the write fails. This is achieved by writing contents to a temporary file
    and moving the temporary file to replace the preexisting file, rather than opening the
    preexisting file for a direct write.

    :param root: Directory name.
    :param file_name: File name. Expects to have '.yaml' extension.
    :param data: The data to write, represented as a dictionary.
    """
    tmp_file_path = None
    try:
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(suffix="file.yaml")
        os.close(tmp_file_fd)
        write_yaml(
            root=get_parent_dir(tmp_file_path),
            file_name=os.path.basename(tmp_file_path),
            data=data,
            overwrite=True,
            sort_keys=True,
        )
        shutil.move(
            tmp_file_path,
            os.path.join(root, file_name),
        )
    finally:
        if tmp_file_path is not None and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def read_yaml(root, file_name):
    """
    Read data from yaml file and return as dictionary

    :param root: Directory name
    :param file_name: File name. Expects to have '.yaml' extension

    :return: Data in yaml file as dictionary
    """
    if not exists(root):
        raise MissingConfigException(
            f"Cannot read '{file_name}'. Parent dir '{root}' does not exist."
        )

    file_path = os.path.join(root, file_name)
    if not exists(file_path):
        raise MissingConfigException("Yaml file '%s' does not exist." % file_path)
    try:
        with codecs.open(file_path, mode="r", encoding=ENCODING) as yaml_file:
            return yaml.load(yaml_file, Loader=YamlSafeLoader)
    except Exception as e:
        raise e


class UniqueKeyLoader(YamlSafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate '{key}' key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def render_and_merge_yaml(root, template_name, context_name):
    """
    Renders a Jinja2-templated YAML file based on a YAML context file, merge them, and return
    result as a dictionary.

    :param root: Root directory of the YAML files
    :param template_name: Name of the template file
    :param context_name: Name of the context file
    :return: Data in yaml file as dictionary
    """
    import jinja2

    template_path = os.path.join(root, template_name)
    context_path = os.path.join(root, context_name)

    for path in (template_path, context_path):
        if not pathlib.Path(path).is_file():
            raise MissingConfigException("Yaml file '%s' does not exist." % path)

    j2_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root, encoding=ENCODING),
        undefined=jinja2.StrictUndefined,
        line_comment_prefix="#",
    )

    def from_json(input_var):
        with open(input_var, encoding="utf-8") as f:
            return json.load(f)

    j2_env.filters["from_json"] = from_json
    # Compute final source of context file (e.g. my-profile.yml), applying Jinja filters
    # like from_json as needed to load context information from files, then load into a dict
    context_source = j2_env.get_template(context_name).render({})
    context_dict = yaml.load(context_source, Loader=UniqueKeyLoader) or {}

    # Substitute parameters from context dict into template
    source = j2_env.get_template(template_name).render(context_dict)
    rendered_template_dict = yaml.load(source, Loader=UniqueKeyLoader)
    return merge_dicts(rendered_template_dict, context_dict)


def read_parquet_as_pandas_df(data_parquet_path: str):
    """
    Deserialize and load the specified parquet file as a Pandas DataFrame.

    :param data_parquet_path: String, path object (implementing os.PathLike[str]),
    or file-like object implementing a binary read() function. The string
    could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
    For file URLs, a host is expected. A local file could
    be: file://localhost/path/to/table.parquet. A file URL can also be a path to a
    directory that contains multiple partitioned parquet files. Pyarrow
    support paths to directories as well as file URLs. A directory
    path could be: file://localhost/path/to/tables or s3://bucket/partition_dir.
    :return: pandas dataframe
    """
    import pandas as pd

    return pd.read_parquet(data_parquet_path, engine="pyarrow")


def write_pandas_df_as_parquet(df, data_parquet_path: str):
    """
    Write a DataFrame to the binary parquet format.

    :param df: pandas data frame.
    :param data_parquet_path: String, path object (implementing os.PathLike[str]),
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
        self._path = os.path.abspath(tempfile.mkdtemp())
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
        assert os.path.exists(os.getcwd())

    def path(self, *path):
        return os.path.join("./", *path) if self._chdr else os.path.join(self._path, *path)


def read_file_lines(parent_path, file_name):
    """
    Return the contents of the file as an array where each element is a separate line.

    :param parent_path: Full path to the directory that contains the file.
    :param file_name: Leaf file name.

    :return: All lines in the file as an array.
    """
    file_path = os.path.join(parent_path, file_name)
    with codecs.open(file_path, mode="r", encoding=ENCODING) as f:
        return f.readlines()


def read_file(parent_path, file_name):
    """
    Return the contents of the file.

    :param parent_path: Full path to the directory that contains the file.
    :param file_name: Leaf file name.

    :return: The contents of the file.
    """
    file_path = os.path.join(parent_path, file_name)
    with codecs.open(file_path, mode="r", encoding=ENCODING) as f:
        return f.read()


def get_file_info(path, rel_path):
    """
    Returns file meta data : location, size, ... etc

    :param path: Path to artifact

    :return: `FileInfo` object
    """
    if is_directory(path):
        return FileInfo(rel_path, True, None)
    else:
        return FileInfo(rel_path, False, os.path.getsize(path))


def get_relative_path(root_path, target_path):
    """
    Remove root path common prefix and return part of `path` relative to `root_path`.

    :param root_path: Root path
    :param target_path: Desired path for common prefix removal

    :return: Path relative to root_path
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
        with gzip.GzipFile(
            filename="", fileobj=open(output_filename, "wb"), mode="wb", mtime=0
        ) as gzipped_tar, open(unzipped_filename, "rb") as tar:
            gzipped_tar.write(tar.read())
    finally:
        os.close(unzipped_file_handle)


def _copy_project(src_path, dst_path=""):
    """
    Internal function used to copy MLflow project during development.

    Copies the content of the whole directory tree except patterns defined in .dockerignore.
    The MLflow is assumed to be accessible as a local directory in this case.


    :param dst_path: MLflow will be copied here
    :return: name of the MLflow project directory
    """

    def _docker_ignore(mlflow_root):
        docker_ignore = os.path.join(mlflow_root, ".dockerignore")
        patterns = []
        if os.path.exists(docker_ignore):
            with open(docker_ignore) as f:
                patterns = [x.strip() for x in f.readlines()]

        def ignore(_, names):
            res = set()
            for p in patterns:
                res.update(set(fnmatch.filter(names, p)))
            return list(res)

        return ignore if patterns else None

    mlflow_dir = "mlflow-project"
    # check if we have project root
    assert os.path.isfile(os.path.join(src_path, "setup.py")), "file not found " + str(
        os.path.abspath(os.path.join(src_path, "setup.py"))
    )
    shutil.copytree(src_path, os.path.join(dst_path, mlflow_dir), ignore=_docker_ignore(src_path))
    return mlflow_dir


def _copy_file_or_tree(src, dst, dst_dir=None):
    """
    :return: The path to the copied artifacts, relative to `dst`
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
    """
    Internal function for reporting the size of a local project directory before copying to
    destination for cli logging reporting to stdout.
    :param project_path: local path of the project directory
    :return: directory file sizes in KB, rounded to single decimal point for legibility
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
            chunk = f.read(chunk_size)
            if chunk:
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


def parallelized_download_file_using_http_uri(
    thread_pool_executor,
    http_uri,
    download_path,
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

    def run_download(range_start, range_end):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = os.path.join(tmpdir, "error_messages.txt")
            download_proc = _exec_cmd(
                cmd=[
                    sys.executable,
                    download_cloud_file_chunk.__file__,
                    "--range-start",
                    range_start,
                    "--range-end",
                    range_end,
                    "--headers",
                    json.dumps(headers or {}),
                    "--download-path",
                    download_path,
                    "--http-uri",
                    http_uri,
                    "--temp-file",
                    temp_file,
                ],
                throw_on_error=True,
                synchronous=False,
                capture_output=True,
                stream_output=False,
                env=env,
            )
            _, stderr = download_proc.communicate()
            if download_proc.returncode != 0:
                if os.path.exists(temp_file):
                    with open(temp_file, "r") as f:
                        file_contents = f.read()
                        if file_contents:
                            return json.loads(file_contents)
                        else:
                            raise Exception(
                                "Error from download_cloud_file_chunk not captured, "
                                f"return code {download_proc.returncode}, stderr {stderr}"
                            )

    num_requests = int(math.ceil(file_size / float(chunk_size)))
    # Create file if it doesn't exist or erase the contents if it does. We should do this here
    # before sending to the workers so they can each individually seek to their respective positions
    # and write chunks without overwriting.
    open(download_path, "w").close()
    starting_index = 0
    if uri_type == ArtifactCredentialType.GCP_SIGNED_URL or uri_type is None:
        # GCP files could be transcoded, in which case the range header is ignored.
        # Test if this is the case by downloading one chunk and seeing if it's larger than the
        # requested size. If yes, let that be the file; if not, continue downloading more chunks.
        download_chunk(
            range_start=0,
            range_end=chunk_size - 1,
            headers=headers,
            download_path=download_path,
            http_uri=http_uri,
        )
        downloaded_size = os.path.getsize(download_path)
        # If downloaded size was equal to the chunk size it would have been downloaded serially,
        # so we don't need to consider this here
        if downloaded_size > chunk_size:
            return {}
        else:
            starting_index = 1

    futures = {}
    for i in range(starting_index, num_requests):
        range_start = i * chunk_size
        range_end = range_start + chunk_size - 1
        futures[thread_pool_executor.submit(run_download, range_start, range_end)] = i

    failed_downloads = {}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                failed_downloads[index] = result

        except Exception as e:
            failed_downloads[index] = {
                "error_status_code": 500,
                "error_text": repr(e),
            }

    return failed_downloads


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
        os.name == "nt"
        and func in (os.unlink, os.rmdir)
        and issubclass(exc_type, PermissionError)
        and exc_value.winerror == 5
    )
    if not should_reattempt:
        raise exc_value
    os.chmod(path, stat.S_IWRITE)
    func(path)


@cache_return_value_per_process
def get_or_create_tmp_dir():
    """
    Get or create a temporary directory which will be removed once python process exit.
    """
    from mlflow.utils.databricks_utils import is_in_databricks_runtime, get_repl_id

    if is_in_databricks_runtime() and get_repl_id() is not None:
        # Note: For python process attached to databricks notebook, atexit does not work.
        # The directory returned by `dbutils.entry_point.getReplLocalTempDir()`
        # will be removed once databricks notebook detaches.
        # The temp directory is designed to be used by all kinds of applications,
        # so create a child directory "mlflow" for storing mlflow temp data.
        try:
            repl_local_tmp_dir = _get_dbutils().entry_point.getReplLocalTempDir()
        except Exception:
            repl_local_tmp_dir = os.path.join("/tmp", "repl_tmp_data", get_repl_id())

        tmp_dir = os.path.join(repl_local_tmp_dir, "mlflow")
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp()
        # mkdtemp creates a directory with permission 0o700
        # change it to be 0o777 to ensure it can be seen in spark UDF
        os.chmod(tmp_dir, 0o777)
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)

    return tmp_dir


@cache_return_value_per_process
def get_or_create_nfs_tmp_dir():
    """
    Get or create a temporary NFS directory which will be removed once python process exit.
    """
    from mlflow.utils.databricks_utils import is_in_databricks_runtime, get_repl_id
    from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir

    nfs_root_dir = get_nfs_cache_root_dir()

    if is_in_databricks_runtime() and get_repl_id() is not None:
        # Note: In databricks, atexit hook does not work.
        # The directory returned by `dbutils.entry_point.getReplNFSTempDir()`
        # will be removed once databricks notebook detaches.
        # The temp directory is designed to be used by all kinds of applications,
        # so create a child directory "mlflow" for storing mlflow temp data.
        try:
            repl_nfs_tmp_dir = _get_dbutils().entry_point.getReplNFSTempDir()
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
    """
    Write spark dataframe in parquet format to local disk.

    :param spark_df: Spark dataframe
    :param output_path: path to write the data to
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
            os.mkdir(abs_dir_path)
        for filename in filenames:
            # For each file with name <filename> immediately under <dirpath>, copy that file to
            # the appropriate location in the destination directory
            file_path = os.path.join(dirpath, filename)
            relative_file_path = os.path.relpath(file_path, src_dir)
            abs_file_path = os.path.join(dst_dir, relative_file_path)
            shutil.copyfile(file_path, abs_file_path)


def contains_path_separator(path):
    """
    Returns True if a path contains a path separator, False otherwise.
    """
    return any((sep in path) for sep in (os.path.sep, os.path.altsep) if sep is not None)


def read_chunk(path: os.PathLike, size: int, start_byte: int = 0) -> bytes:
    """
    Read a chunk of bytes from a file.

    :param path: Path to the file.
    :param size: The size of the chunk.
    :param start_byte: The start byte of the chunk.
    :return: The chunk of bytes.
    """
    with open(path, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
        return f.read(size)


@contextmanager
def remove_on_error(path: os.PathLike, onerror=None):
    """
    A context manager that removes a file or directory if an exception is raised during execution.

    :param path: Path to the file or directory.
    :param onerror: A callback function that will be called with the captured exception before
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
        raise
