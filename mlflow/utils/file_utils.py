import codecs
import gzip
import os
import posixpath
import shutil
import sys
import tarfile
import tempfile

from six.moves.urllib.request import pathname2url
from six.moves.urllib.parse import unquote
from six.moves import urllib

import yaml

from mlflow.entities import FileInfo
from mlflow.exceptions import MissingConfigException

ENCODING = "utf-8"


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


def mkdir(root, name=None):  # noqa
    """
    Make directory with name "root/name", or just "root" if name is None.

    :param root: Name of parent directory
    :param name: Optional name of leaf directory

    :return: Path to created directory
    """
    target = os.path.join(root, name) if name is not None else root
    try:
        if not exists(target):
            os.makedirs(target)
            return target
    except OSError as e:
        raise e


def make_containing_dirs(path):
    """
    Create the base directory for a given file path if it does not exist; also creates parent
    directories.
    """
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_yaml(root, file_name, data, overwrite=False):
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
        raise Exception("Yaml file '%s' exists as '%s" % (file_path, yaml_file_name))

    try:
        with codecs.open(yaml_file_name, mode='w', encoding=ENCODING) as yaml_file:
            yaml.safe_dump(data, yaml_file, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise e


def read_yaml(root, file_name):
    """
    Read data from yaml file and return as dictionary

    :param root: Directory name
    :param file_name: File name. Expects to have '.yaml' extension

    :return: Data in yaml file as dictionary
    """
    if not exists(root):
        raise MissingConfigException(
            "Cannot read '%s'. Parent dir '%s' does not exist." % (file_name, root))

    file_path = os.path.join(root, file_name)
    if not exists(file_path):
        raise MissingConfigException("Yaml file '%s' does not exist." % file_path)
    try:
        with codecs.open(file_path, mode='r', encoding=ENCODING) as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e


class TempDir(object):
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
    with codecs.open(file_path, mode='r', encoding=ENCODING) as f:
        return f.readlines()


def read_file(parent_path, file_name):
    """
    Return the contents of the file.

    :param parent_path: Full path to the directory that contains the file.
    :param file_name: Leaf file name.

    :return: The contents of the file.
    """
    file_path = os.path.join(parent_path, file_name)
    with codecs.open(file_path, mode='r', encoding=ENCODING) as f:
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
        raise Exception("Root path '%s' longer than target path '%s'" % (root_path, target_path))
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

    unzipped_filename = tempfile.mktemp()
    try:
        with tarfile.open(unzipped_filename, "w") as tar:
            tar.add(source_dir, arcname=archive_name, filter=_filter_timestamps)
        # When gzipping the tar, don't include the tar's filename or modification time in the
        # zipped archive (see https://docs.python.org/3/library/gzip.html#gzip.GzipFile)
        with gzip.GzipFile(filename="", fileobj=open(output_filename, 'wb'), mode='wb', mtime=0) \
                as gzipped_tar, open(unzipped_filename, 'rb') as tar:
            gzipped_tar.write(tar.read())
    finally:
        os.remove(unzipped_filename)


def _copy_project(src_path, dst_path=""):
    """
    Internal function used to copy MLflow project during development.

    Copies the content of the whole directory tree except patterns defined in .dockerignore.
    The MLflow is assumed to be accessible as a local directory in this case.


    :param dst_path: MLflow will be copied here
    :return: name of the MLflow project directory
    """

    def _docker_ignore(mlflow_root):
        docker_ignore = os.path.join(mlflow_root, '.dockerignore')
        patterns = []
        if os.path.exists(docker_ignore):
            with open(docker_ignore, "r") as f:
                patterns = [x.strip() for x in f.readlines()]

        def ignore(_, names):
            import fnmatch
            res = set()
            for p in patterns:
                res.update(set(fnmatch.filter(names, p)))
            return list(res)

        return ignore if patterns else None

    mlflow_dir = "mlflow-project"
    # check if we have project root
    assert os.path.isfile(os.path.join(src_path, "setup.py")), "file not found " + str(
        os.path.abspath(os.path.join(src_path, "setup.py")))
    shutil.copytree(src_path, os.path.join(dst_path, mlflow_dir),
                    ignore=_docker_ignore(src_path))
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
        shutil.copytree(src=src, dst=dst_path)
    return dst_subpath


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
    path = pathname2url(path)
    if path == posixpath.abspath(path):
        return "file://{path}".format(path=path)
    else:
        return "file:{path}".format(path=path)


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
    path = urllib.parse.urlparse(uri).path if uri.startswith("file:") else uri
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
