import codecs
import json
import os
import pathlib
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
from mlflow.utils import merge_dicts


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
    """Renders a Jinja2-templated YAML file based on a YAML context file, merge them, and return
    result as a dictionary.

    Args:
        root: Root directory of the YAML files.
        template_name: Name of the template file.
        context_name: Name of the context file.

    Returns:
        Data in yaml file as dictionary.
    """
    from jinja2 import FileSystemLoader, StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment

    template_path = os.path.join(root, template_name)
    context_path = os.path.join(root, context_name)

    for path in (template_path, context_path):
        if not pathlib.Path(path).is_file():
            raise MissingConfigException(f"Yaml file '{path}' does not exist.")

    j2_env = SandboxedEnvironment(
        loader=FileSystemLoader(root, encoding=ENCODING),
        undefined=StrictUndefined,
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
