# Databricks CLI
# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"), except
# that the use of services to which certain application programming
# interfaces (each, an "API") connect requires that the user first obtain
# a license for the use of the APIs from Databricks, Inc. ("Databricks"),
# by creating an account at www.databricks.com and agreeing to either (a)
# the Community Edition Terms of Service, (b) the Databricks Terms of
# Service, or (c) another written agreement between Licensee and Databricks
# for the use of the APIs.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        'awscli',
        'click>=6.7',
        'databricks-cli',
        'requests>=2.17.3',
        'six>=1.10.0',
        'uuid',
        'gitpython',
        'Flask',
        'pygal',
        'zipstream',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'python-dateutil',
        'protobuf',
        'gitpython',
        'pyyaml',
        'boto3',
        'querystring_parser',
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
