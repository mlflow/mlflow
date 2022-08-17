# MLflow Development Container

This directory contains a set of files to automatically build a development environment for MLflow on Visual Studio Code using [the Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Supported Features

- Pre-installed development tools.
- Auto formatting and lint check for Python code.
- Auto formatting and lint check for JavaScript code.
- Pre-commit hook to prevent pushing commits that won't pass the CI checks.
- Useful Visual Studio Code extensions to increase productivity.

## Prerequisites

- Visual Studio Code
- The Remote - Containers extension
- Docker

See [Remote development in Containers](https://code.visualstudio.com/docs/remote/containers-tutorial) for more information.

## Getting Started

1. Open the repository in Visual Studio Code.
2. Open [the command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette) and select `Remote-Containers: Reopen in Container`.
3. Once the devcontainer is built and running, run the following commands to ensure the development environment is set up properly:

```bash
python examples/sklearn_autolog/linear_regression.py
pytest tests/test_version.py
./dev/run-dev-server
```

See [Developing inside a Container](https://code.visualstudio.com/docs/remote/containers) for more information.

## Developing in a GitHub Codespaces

You can also launch the devcontainer using [GitHub Codespaces](https://github.com/features/codespaces). See [Developing in codespaces](https://docs.github.com/en/codespaces/developing-in-codespaces) for more information.

## Rebuild devcontainer

1. Run `docker pull ghcr.io/mlflow/mlflow-devcontainer`.
2. Open the command palette and run `Remote-Containers: Rebuild Container`.

## Limitations

The devcontainer image doesn't contain extra ML packages such as `tensorflow` to reduce the image size, but they can be installed after launching the devcontainer:

```
pip install tensorflow
```
