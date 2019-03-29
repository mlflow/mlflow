import yaml

from mlflow.utils import PYTHON_VERSION

_conda_header = """\
name: mlflow-env
channels:
  - defaults
"""

DEFAULT_PIP_DEPENDENCIES = ['gunicorn',
                        'docker>=3.6.0',
                        'entrypoints',
                        'protobuf>=3.6.0',
                        'numpy',
                        'pandas',
                        'pyyaml',
                        'boto3',
                        'click',
                        'Flask',
                        'databricks-cli>=0.8.0',
                        'querystring_parser',
                        'sqlparse',
                        ]


def _mlflow_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None,
                      additional_conda_channels=None):
    """
    Creates a Conda environment with the specified package channels and dependencies.

    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in dictionary format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_channels: List of additional conda channels to search when resolving packages.
    :return: `None` if `path` is specified. Otherwise, the a dictionary representation of the
             Conda environment.
    """
    env = yaml.load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    if additional_pip_deps is not None:
        env["dependencies"].append({"pip": additional_pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def _prepare_dependency_map(dependencies):
    """
    Helper method to get the dictionary of package name and dependency
    :param dependencies: list of packages
    :return: dictionary with key as package name and value with package name along with version if any
    Input:['gunicorn', 'docker>=3.6.0', 'protobuf>=3.6.0', 'cloudpickle==0.6.1', 'python=3.6.1']
    Output: {'gunicorn': 'gunicorn', 'docker': 'docker>=3.6.0', 'protobuf': 'protobuf>=3.6.0',
                'cloudpickle': 'cloudpickle==0.6.1',  'python': 'python=3.6.1'}
    """
    dependency_map = {}
    for dependency in dependencies:
        if '>=' in dependency:
            package_name, _ = dependency.split('>=')
        elif '==' in dependency:
            package_name, _ = dependency.split('==')
        elif '=' in dependency:
            package_name, _ = dependency.split('=')
        else:
            package_name = dependency
        dependency_map[package_name] = dependency
    return dependency_map


def update_conda_env_deps(conda_env, pip_dependencies):
    """
    Updates the conda env with providing pip_dependencies while checking for existing ones in conda env.
    If same package is provided in conda_env and pip_dependencies , package in conda_env will be preceded
    and the one in pip_dependencies will be ignored
    :param conda_env: dict of conda yaml file
    :param pip_dependencies: list of pip packages
    :return: Updated conda env dict
    Input:
        conda_env=
                    {'name': 'mlflow-env',
                    'channels': ['defaults'],
                    'dependencies':  [
                                'python=3.6.1',
                                'scikit-learn=0.19.1',
                                'gunicorn=19.8.0',
                                {'pip':
                                    ['docker>=3.5.0',
                                    'entrypoints'
                                    ]}]}

        pip_dependencies = [
                    'gunicorn==19.8.1',
                    'docker>=3.6.0',
                    'entrypoints',
                    'protobuf>=3.6.0',
                    'numpy',
                    'pandas',
                    'pyyaml']
    Output:
            {'name': 'mlflow-env',
             'channels': ['defaults'],
             'dependencies': [
                    'python=3.6.1',
                    'scikit-learn=0.19.1',
                    'gunicorn=19.8.0',
                    {'pip':
                        [
                            'docker>=3.5.0',
                            'entrypoints',
                            'protobuf>=3.6.0',
                            'numpy',
                            'pandas',
                            'pyyaml'
                        ]}]}

    """
    if conda_env:
        dependencies = conda_env.get('dependencies')
        if dependencies:
            env_pip_dependencies = []
            if isinstance(dependencies[-1], dict):
                env_pip_dependencies = dependencies[-1]['pip']
            else:
                dependencies.append({"pip": env_pip_dependencies})
            # packages along with versions
            conda_env_all_packages = dependencies[:-1] + env_pip_dependencies
            conda_env_all_package_map = _prepare_dependency_map(conda_env_all_packages)
            # package names without versions
            conda_env_all_package_names = set(conda_env_all_package_map.keys())
            pip_dependency_map = _prepare_dependency_map(pip_dependencies)
            for package_name, dependency in pip_dependency_map.items():
                if package_name not in conda_env_all_package_names:
                    env_pip_dependencies.append(dependency)
    return conda_env
