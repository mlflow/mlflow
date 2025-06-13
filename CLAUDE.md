# MLflow Development Guide

## Dependency versions
- MLflow uses **Python 3.10**
- ML package dependencies are listed in ml-package-versions.yaml file. Note that you MUST NOT install all libraries defined in this file, which will cause hell of dependency conflict. You should only install library that is required for your task.
- The virtual environment for development including these are available at `.venv`. You must activate the environment before running tests, linter, etc.
 
## Build/Lint/Test Commands
- **Activate virtual environment**: `source ~/Workspace/claude-workspace/mlflow/.venv/bin/activate``
- **Run a single test**: `pytest tests/path/to/test_file.py::test_function_name -v`
- **Run all tests**: `pytest tests --quiet --requires-ssh --ignore-flavors --ignore=tests/examples --ignore=tests/recipes --ignore=tests/evaluate`
- **Run specific module tests**: `pytest tests/tracking/test_client.py`
- **Run pre-commit hooks for linting and format checks**: `pre-commit run --all-files`
- **Build docs**: `cd docs && yarn && yarn start`
- **Build UI dev server**: `cd mlflow/server/js && yarn start`
- **Start mlflow server**: `mlflow ui`

## Code Style Guidelines
- **Python Style**: Follow Google's Python Style Guide for docstrings and general conventions
- **Code Formatting**: Use ruff for auto-formatting
- **Imports**: Ban relative imports (`from mlflow import X` instead of `from . import X`)
- **Types**: Use type annotations and avoid deprecated numpy type aliases
- **Naming Conventions**: Follow Python standards (snake_case for functions/variables, CamelCase for classes)
- **Error Handling**: Write proper error messages and use appropriate exception types
- **Tests**: Write tests for all new features, ensure backward compatibility
- **Backward Compatibility**: Carefully maintain API compatibility for users

## Important Notes
- Sign your commits with `git commit -s` (required for contributions)
- When working on a new change, you must cut a feature branch and edit change there. The feature branch must be cut from the up-to-date master branch. Run `git pull databricks master` to update master branch and cut a feature branch from there.
- For UI development, use JavaScript dev server

