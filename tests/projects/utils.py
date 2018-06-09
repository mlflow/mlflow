import os

import yaml

from mlflow.projects import Project

TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")
GIT_PROJECT_URI = "https://github.com/databricks/mlflow-example"


def load_project():
    """ Loads an example project for use in tests, returning an in-memory `Project` object. """
    with open(os.path.join(TEST_PROJECT_DIR, "MLproject")) as mlproject_file:
        project_yaml = yaml.safe_load(mlproject_file.read())
    return Project(uri=TEST_PROJECT_DIR, yaml_obj=project_yaml)
