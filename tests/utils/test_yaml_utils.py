import codecs
import os

from mlflow.utils.yaml_utils import (
    read_yaml,
    safe_edit_yaml,
    write_yaml,
)

from tests.helper_functions import random_file, random_int


def test_yaml_read_and_write(tmp_path):
    temp_dir = str(tmp_path)
    yaml_file = random_file("yaml")
    long_value = 1
    data = {
        "a": random_int(),
        "B": random_int(),
        "text_value": "中文",
        "long_value": long_value,
        "int_value": 32,
        "text_value_2": "hi",
    }
    write_yaml(temp_dir, yaml_file, data)
    read_data = read_yaml(temp_dir, yaml_file)
    assert data == read_data
    yaml_path = os.path.join(temp_dir, yaml_file)
    with codecs.open(yaml_path, encoding="utf-8") as handle:
        contents = handle.read()
    assert "!!python" not in contents
    # Check that UTF-8 strings are written properly to the file (rather than as ASCII
    # representations of their byte sequences).
    assert "中文" in contents

    def edit_func(old_dict):
        old_dict["more_text"] = "西班牙语"
        return old_dict

    assert "more_text" not in read_yaml(temp_dir, yaml_file)
    with safe_edit_yaml(temp_dir, yaml_file, edit_func):
        edited_dict = read_yaml(temp_dir, yaml_file)
        assert "more_text" in edited_dict
        assert edited_dict["more_text"] == "西班牙语"
    assert "more_text" not in read_yaml(temp_dir, yaml_file)


def test_yaml_write_sorting(tmp_path):
    temp_dir = str(tmp_path)
    data = {
        "a": 1,
        "c": 2,
        "b": 3,
    }

    sorted_yaml_file = random_file("yaml")
    write_yaml(temp_dir, sorted_yaml_file, data, sort_keys=True)
    expected_sorted = """a: 1
b: 3
c: 2
"""
    with open(os.path.join(temp_dir, sorted_yaml_file)) as f:
        actual_sorted = f.read()

    assert actual_sorted == expected_sorted

    unsorted_yaml_file = random_file("yaml")
    write_yaml(temp_dir, unsorted_yaml_file, data, sort_keys=False)
    expected_unsorted = """a: 1
c: 2
b: 3
"""
    with open(os.path.join(temp_dir, unsorted_yaml_file)) as f:
        actual_unsorted = f.read()

    assert actual_unsorted == expected_unsorted
