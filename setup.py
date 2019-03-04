import imp
import os
from setuptools import setup, find_packages

version = imp.load_source(
    'mlflow.version', os.path.join('mlflow', 'version.py')).VERSION


# Get a list of all files in the JS directory to include in our module
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


# Prints out a set of paths (relative to the mlflow/ directory) of files in mlflow/server/js/build
# to include in the wheel, e.g. "../mlflow/server/js/build/index.html"
js_files = package_files('mlflow/server/js/build')
sagmaker_server_files = package_files("mlflow/sagemaker/container")

setup(
    name='mlflow',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={"mlflow": js_files + sagmaker_server_files},
    install_requires=[
        'click>=7.0',
        'cloudpickle==0.6.1',
        'databricks-cli>=0.8.0',
        'requests>=2.17.3',
        'six>=1.10.0',
        'gunicorn',
        'Flask',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'python-dateutil',
        'protobuf>=3.6.0',
        'gitpython>=2.1.0',
        'pyyaml',
        'boto3>=1.7.12',
        'querystring_parser',
        'simplejson',
        'mleap>=0.8.1',
        'cloudpickle',
        'docker>=3.6.0',
        'entrypoints',
        'sqlparse',
    ],
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ml ai databricks',
    url='https://mlflow.org/'
)
