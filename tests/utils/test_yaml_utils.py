import codecs
import json
import os

import jinja2.exceptions
import pytest

from mlflow.exceptions import MissingConfigException
from mlflow.utils.yaml_utils import (
    read_yaml,
    render_and_merge_yaml,
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


def test_render_and_merge_yaml(tmp_path, monkeypatch):
    json_file = random_file("json")
    extra_config = {"key": 123}
    with open(tmp_path / json_file, "w") as f:
        json.dump(extra_config, f)

    template_yaml_file = random_file("yaml")
    with open(tmp_path / template_yaml_file, "w") as f:
        f.write(
            """
            steps:
              preprocess:
                train_ratio: {{ MY_TRAIN_RATIO|default(0.5) }}
                experiment:
                  tracking_uri: {{ MY_MLFLOW_SERVER|default("https://localhost:5000") }}
            test_1: [1, 2, 3]
            test_2: {{ TEST_VAR_1 }}
            test_3: {{ TEST_VAR_2 }}
            test_4: {{ TEST_VAR_4 }}
            """
            + rf"test_5: {{{{ ('{json_file}' | from_json)['key'] }}}}"
        )
    context_yaml_file = random_file("yaml")
    with open(tmp_path / context_yaml_file, "w") as f:
        f.write(
            """
            MY_MLFLOW_SERVER: "./mlruns"
            TEST_VAR_1: ["a", 1.2]
            TEST_VAR_2: {"a": 2}
            """
            + rf"TEST_VAR_4: {{{{ ('{json_file}' | from_json)['key'] }}}}"
        )

    monkeypatch.chdir(tmp_path)
    result = render_and_merge_yaml(tmp_path, template_yaml_file, context_yaml_file)
    expected = {
        "MY_MLFLOW_SERVER": "./mlruns",
        "TEST_VAR_1": ["a", 1.2],
        "TEST_VAR_2": {"a": 2},
        "TEST_VAR_4": 123,
        "steps": {"preprocess": {"train_ratio": 0.5, "experiment": {"tracking_uri": "./mlruns"}}},
        "test_1": [1, 2, 3],
        "test_2": ["a", 1.2],
        "test_3": {"a": 2},
        "test_4": 123,
        "test_5": 123,
    }
    assert result == expected


def test_render_and_merge_yaml_raise_on_duplicate_keys(tmp_path):
    template_yaml_file = random_file("yaml")
    with open(tmp_path / template_yaml_file, "w") as f:
        f.write(
            """
            steps: 1
            steps: 2
            test_2: {{ TEST_VAR_1 }}
            """
        )

    context_yaml_file = random_file("yaml")
    write_yaml(str(tmp_path), context_yaml_file, {"TEST_VAR_1": 3})

    with pytest.raises(ValueError, match="Duplicate 'steps' key found"):
        render_and_merge_yaml(tmp_path, template_yaml_file, context_yaml_file)


def test_render_and_merge_yaml_raise_on_non_existent_yamls(tmp_path):
    template_yaml_file = random_file("yaml")
    with open(tmp_path / template_yaml_file, "w") as f:
        f.write("""test_1: {{ TEST_VAR_1 }}""")

    context_yaml_file = random_file("yaml")
    write_yaml(str(tmp_path), context_yaml_file, {"TEST_VAR_1": 3})

    with pytest.raises(MissingConfigException, match="does not exist"):
        render_and_merge_yaml(tmp_path, "invalid_name", context_yaml_file)
    with pytest.raises(MissingConfigException, match="does not exist"):
        render_and_merge_yaml("invalid_path", template_yaml_file, context_yaml_file)
    with pytest.raises(MissingConfigException, match="does not exist"):
        render_and_merge_yaml(tmp_path, template_yaml_file, "invalid_name")


def test_render_and_merge_yaml_raise_on_not_found_key(tmp_path):
    template_yaml_file = random_file("yaml")
    with open(tmp_path / template_yaml_file, "w") as f:
        f.write("""test_1: {{ TEST_VAR_1 }}""")

    context_yaml_file = random_file("yaml")
    write_yaml(str(tmp_path), context_yaml_file, {})

    with pytest.raises(jinja2.exceptions.UndefinedError, match="'TEST_VAR_1' is undefined"):
        render_and_merge_yaml(tmp_path, template_yaml_file, context_yaml_file)


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
