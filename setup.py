import os
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

version = SourceFileLoader(
    'mlflow.version', os.path.join('mlflow', 'version.py')).load_module().VERSION


# Get a list of all files in the JS directory to include in our module
def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


# Prints out a set of paths (relative to the mlflow/ directory) of files in mlflow/server/js/build
# to include in the wheel, e.g. "../mlflow/server/js/build/index.html"
js_files = package_files('mlflow/server/js/build')
models_container_server_files = package_files("mlflow/models/container")
alembic_files = ["../mlflow/store/db_migrations/alembic.ini",
                 "../mlflow/temporary_db_migrations_for_pre_1_users/alembic.ini"]

setup(
    name='mlflow',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={"mlflow": js_files + models_container_server_files + alembic_files},
    install_requires=[
        'alembic',
        'azure-storage-blob>=12.0',
        'click>=7.0',
        'cloudpickle',
        'databricks-cli>=0.8.7',
        'requests>=2.17.3',
        'six>=1.10.0',
        'waitress; platform_system == "Windows"',
        'gunicorn; platform_system != "Windows"',
        'Flask',
        'numpy',
        'pandas',
        'python-dateutil',
        'protobuf>=3.6.0',
        'gitpython>=2.1.0',
        'pyyaml',
        'querystring_parser',
        'docker>=4.0.0',
        'entrypoints',
        'sqlparse',
        'sqlalchemy<=1.3.13',
        'gorilla',
        'prometheus-flask-exporter',
    ],
    extras_require={
        'extras': [
            "scikit-learn; python_version >= '3.5'",
            # scikit-learn 0.20 is the last version to support Python 2.x  & Python 3.4.
            "scikit-learn==0.20; python_version < '3.5'",
            'boto3>=1.7.12',
            'mleap>=0.16.0',
            'azure-storage-blob>=12.0',
            'google-cloud-storage',
            'azureml-core>=1.2.0'
        ],
         'sqlserver': [
            "mlflow-dbstore",
        ],
         'aliyun-oss': [
            "aliyunstoreplugin",
        ],
    },
    entry_points='''
        [console_scripts]
        mlflow=mlflow.cli:cli
    ''',
    zip_safe=False,
    author='Databricks',
    description='MLflow: An ML Workflow Tool',
    long_description=open('README.rst').read(),
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ml ai databricks',
    url='https://mlflow.org/',
    python_requires='>=3.5',
    project_urls={
        'Bug Tracker': 'https://github.com/mlflow/mlflow/issues',
        'Documentation': 'https://mlflow.org/docs/latest/index.html',
        'Source Code': 'https://github.com/mlflow/mlflow'
    },
)
