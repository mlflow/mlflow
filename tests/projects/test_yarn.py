import pytest
import textwrap

import skein

from mlflow.projects import _project_spec
from mlflow.exceptions import ExecutionException
from mlflow.projects.yarn import _validate_yarn_env, _generate_skein_service, \
    _get_key_from_params, _parse_yarn_config
from tests.projects.utils import TEST_YARN_PROJECT_DIR


def test_valid_project_backend_yarn():
    project = _project_spec.load_project(TEST_YARN_PROJECT_DIR)
    _validate_yarn_env(project)


def test_invalid_project_backend_yarn():
    project = _project_spec.load_project(TEST_YARN_PROJECT_DIR)
    project.name = None
    with pytest.raises(ExecutionException):
        _validate_yarn_env(project)


def test_get_key_from_params():
    extra_params = {
        'env': 'ENV1=ENV1,ENV2=ENV2',
        'additional_files': '/user/myfile1.py,/user/myfile2.py'
    }
    env = _get_key_from_params(extra_params, 'env')
    additional_files = _get_key_from_params(extra_params, 'additional_files')
    empty_param = _get_key_from_params(extra_params, 'no_key')
    assert env == ['ENV1=ENV1', 'ENV2=ENV2']
    assert additional_files == ['/user/myfile1.py', '/user/myfile2.py']
    assert empty_param == []


def test_generate_skein_service():
    expected_command = textwrap.dedent("""
                 set -x
                 env
                 export HADOOP_CONF_DIR=/etc/hadoop/conf
                 export PATH=$(pwd)/conda_env.zip/bin:$PATH
                 export LD_LIBRARY_PATH=$(pwd)/conda_env.zip/lib
                 cd project.zip
                 python train.py
             """)
    expected_files = {
        'test.py': skein.model.File('test.py')
    }

    service = _generate_skein_service("1 GiB", 1, 1, {"ENV": "ENV"}, ["test.py"],
                                      "/etc/hadoop/conf", "python train.py")

    assert service.instances == 1
    assert service.env['ENV'] == 'ENV'
    assert service.script == expected_command
    assert service.resources == skein.model.Resources("1 GiB", 1)
    assert service.files == expected_files


def test_parse_yarn_config():
    backend_config = {
        "additional_files": ["./notebooks/specialMessage.ipynb"],
        "num_cores": 12,
        "memory": 2048,
        "queue": "default",
        "hadoop_filesystems": "viewfs://filesystem",
        "hadoop_conf_dir": "/etc/hadoop/conf"
    }
    expected_backend_config = {
        "additional_files": ["./notebooks/specialMessage.ipynb"],
        "num_cores": 12,
        "memory": 2048,
        "queue": "default",
        "hadoop_filesystems": "viewfs://filesystem",
        "hadoop_conf_dir": "/etc/hadoop/conf",
        "env": {}
    }
    yarn_config = _parse_yarn_config(backend_config)
    assert yarn_config == expected_backend_config

    extra_params = {
        'env': 'env'
    }
    yarn_config = _parse_yarn_config(backend_config, extra_params)
    expected_backend_config['env'] = 'env'
    assert yarn_config == expected_backend_config
