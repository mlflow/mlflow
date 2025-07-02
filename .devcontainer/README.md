# MLflow development container (experimental)

This directory contains a set of files to set up a reproducible and disposable development environment for MLflow in Visual Studio Code using the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Supported features

- Pre-installed tools/packages required for MLflow development.
- Pre-configured VSCode settings and extensions for automatic code formatting and lint check.
- Pre-commit hooks to avoid pushing commits that won't pass the CI checks.

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/)

## Getting started

1. Build the devcontainer image.

   ```bash
   # This command may take a while to finish
   DOCKER_BUILDKIT=1 docker build -f .devcontainer/Dockerfile.devcontainer -t ghcr.io/mlflow/mlflow-devcontainer .

   # Test the image
   docker run --rm -it ghcr.io/mlflow/mlflow-devcontainer zsh

   # Alternatively, you can pull the pre-built image from GitHub Container Registry,
   # but a GitHub personal access token is required to authenticate to ghcr.io:
   echo <GITHUB_TOKEN> | docker login ghcr.io -u <GITHUB_USERNAME> --password-stdin
   docker pull ghcr.io/mlflow/mlflow-devcontainer
   ```

2. Open the MLflow repository on VSCode.
3. Press `Ctrl/Cmd+Shift+P` to launch [the command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette).
4. Select `Remote-Containers: Reopen in Container`.
5. Once the devcontainer is up and running, launch the command palette again.
6. Select `Terminal: Create New Terminal`.
7. Run the following commands and make sure they run successfully:

```bash
pytest tests/test_version.py
```

## Developing in GitHub Codespaces

You can create the same development environment as your local devcontainer **in a web browser with just a few clicks** using [GitHub Codespaces](https://github.com/features/codespaces). The instructions in [Creating a codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace#creating-a-codespace) cover how to set up a codespace.

<img src="./images/codespace.png" width="50%">

## Limitations

The following tools/packages are NOT pre-installed to avoid increasing the image size. They can be manually installed after launching the devcontainer if necessary.

- Python ML packages such as `tensorflow`
- R
- Docker
- conda

## How to update `requirements.txt`

```bash
NAME="mlflow-$(uuidgen)"
docker run \
   --name $NAME \
   -w /mlflow \
   -v $(pwd)/requirements:/mlflow/requirements:ro \
   -v $(pwd)/.devcontainer/pip-compile.sh:/mlflow/pip-compile.sh \
   python:3.10.16 ./pip-compile.sh
docker cp $NAME:/tmp/requirements.txt .devcontainer/requirements.txt
docker rm -f -v $NAME
```
