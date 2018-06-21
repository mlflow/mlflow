import yaml

from mlflow.utils import PYTHON_VERSION
from mlflow.version import VERSION as MLFLOW_VERSION


class CondaEnvironment(object):
    def __init__(self, name="mlflow_env", channels=('anaconda', 'defaults')):
        self.name = name
        self.channels = channels
        self.dependencies = {'python': PYTHON_VERSION, "pip": {"mlflow": MLFLOW_VERSION}}

    def save(self, path):
        with open(path, 'w') as out:
            yaml.safe_dump(self.__dict__, stream=out, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**yaml.safe_load(f.read()))

    def add_conda_dependency(self, package_name, version=None):
        _add_dependency(self.dependencies, package_name, version)

    def add_pip_dependency(self, package_name, version=None):
        _add_dependency(self.dependencies['pip'], package_name, version)


def _add_dependency(dependencies, package_name, version):
    if package_name in dependencies:
        current_version = dependencies[package_name]
        if version != current_version:
            raise Exception("Already has a dependency on {package}=={version}".format(
                package=package_name, version=current_version
            ))
    else:
        dependencies[package_name] = version
