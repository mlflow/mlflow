import codecs
import os
import shutil
import tempfile

import yaml

from mlflow.utils.file_utils import ENCODING, exists, get_parent_dir

try:
    from yaml import CSafeDumper as YamlSafeDumper
    from yaml import CSafeLoader as YamlSafeLoader

except ImportError:
    from yaml import SafeDumper as YamlSafeDumper
    from yaml import SafeLoader as YamlSafeLoader

from mlflow.exceptions import MissingConfigException


def write_yaml(root, file_name, data, overwrite=False, sort_keys=True, ensure_yaml_extension=True):
    """Write dictionary data in yaml format.

    Args:
        root: Directory name.
        file_name: Desired file name.
        data: Data to be dumped as yaml format.
        overwrite: If True, will overwrite existing files.
        sort_keys: Whether to sort the keys when writing the yaml file.
        ensure_yaml_extension: If True, will automatically add .yaml extension if not given.
    """
    if not exists(root):
        raise MissingConfigException(f"Parent directory '{root}' does not exist.")

    file_path = os.path.join(root, file_name)
    yaml_file_name = file_path
    if ensure_yaml_extension and not file_path.endswith(".yaml"):
        yaml_file_name = file_path + ".yaml"

    if exists(yaml_file_name) and not overwrite:
        raise Exception(f"Yaml file '{file_path}' exists as '{yaml_file_name}")

    with codecs.open(yaml_file_name, mode="w", encoding=ENCODING) as yaml_file:
        yaml.dump(
            data,
            yaml_file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys,
            Dumper=YamlSafeDumper,
        )


def overwrite_yaml(root, file_name, data, ensure_yaml_extension=True):
    """Safely overwrites a preexisting yaml file, ensuring that file contents are not deleted or
    corrupted if the write fails. This is achieved by writing contents to a temporary file
    and moving the temporary file to replace the preexisting file, rather than opening the
    preexisting file for a direct write.

    Args:
        root: Directory name.
        file_name: File name.
        data: The data to write, represented as a dictionary.
        ensure_yaml_extension: If True, Will automatically add .yaml extension if not given.
    """
    tmp_file_path = None
    original_file_path = os.path.join(root, file_name)
    original_file_mode = os.stat(original_file_path).st_mode
    try:
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(suffix="file.yaml")
        os.close(tmp_file_fd)
        write_yaml(
            root=get_parent_dir(tmp_file_path),
            file_name=os.path.basename(tmp_file_path),
            data=data,
            overwrite=True,
            sort_keys=True,
            ensure_yaml_extension=ensure_yaml_extension,
        )
        shutil.move(tmp_file_path, original_file_path)
        # restores original file permissions, see https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp
        os.chmod(original_file_path, original_file_mode)
    finally:
        if tmp_file_path is not None and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def read_yaml(root, file_name):
    """Read data from yaml file and return as dictionary
    Args:
        root: Directory name.
        file_name: File name. Expects to have '.yaml' extension.

    Returns:
        Data in yaml file as dictionary.
    """
    if not exists(root):
        raise MissingConfigException(
            f"Cannot read '{file_name}'. Parent dir '{root}' does not exist."
        )

    file_path = os.path.join(root, file_name)
    if not exists(file_path):
        raise MissingConfigException(f"Yaml file '{file_path}' does not exist.")
    with codecs.open(file_path, mode="r", encoding=ENCODING) as yaml_file:
        return yaml.load(yaml_file, Loader=YamlSafeLoader)


class safe_edit_yaml:
    def __init__(self, root, file_name, edit_func):
        self._root = root
        self._file_name = file_name
        self._edit_func = edit_func
        self._original = read_yaml(root, file_name)

    def __enter__(self):
        new_dict = self._edit_func(self._original.copy())
        write_yaml(self._root, self._file_name, new_dict, overwrite=True)

    def __exit__(self, *args):
        write_yaml(self._root, self._file_name, self._original, overwrite=True)
