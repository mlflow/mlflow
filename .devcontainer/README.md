# MLflow Development Container

This directory contains files to automatically build an out-of-the-box development environment for MLflow in Visual Studio Code using [the Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Supported features

- Pre-installed development packages
- Auto formatting for Python code
- Auto formatting and lint check for JavaScript code
- Auto formatting for the following file types via `prettier`
  - `.md`
  - `.yaml`
  - `.json`
- Pre-commit hook to prevent pushing commits that won't pass the CI checks
- Useful Visual Studio Code extensions to increase productivity
- Zsh terminal with a [Powerlevel10k](https://github.com/romkatv/powerlevel10k) theme

## Prerequisites

See [Remote development in Containers](https://code.visualstudio.com/docs/remote/containers-tutorial) and set up the following tools:

- Visual Studio Code
- The `Remote - Containers` extension
- Docker

## Getting started

1. Open the repository in Visual Studio Code.
2. Open [the command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette) and run `Remote-Containers: Reopen in Container`. This step may not finish quickly because the devcontainer image is large and it takes a while to pull it from GitHub Container Registry.
3. Once the devcontainer is built and running, run the following commands to make sure the devcontainer is set up properly:

```bash
python examples/sklearn_autolog/linear_regression.py
./dev/run-dev-server
```

### See also:

- [Developing inside a Container](https://code.visualstudio.com/docs/remote/containers)

## Developing in a GitHub Codespace

You can also launch the devcontainer using [GitHub Codespaces](https://github.com/features/codespaces). See [Developing in a codespace](https://docs.github.com/en/codespaces/developing-in-codespaces) for details.

### See also:

- [Settings Sync in Visual Studio Code](https://code.visualstudio.com/docs/editor/settings-sync)
- [Personalizing Codespaces for your account](https://docs.github.com/en/codespaces/customizing-your-codespace/personalizing-codespaces-for-your-account)

## Rebuild devcontainer

1. Run `docker pull ghcr.io/mlflow/mlflow-devcontainer`.
2. Open the command palette and run `Remote-Containers: Rebuild Container`.

## Limitations

Currently, the devcontainer image doesn't contain extra ML packages (excluding `scikit-learn`) because installing them significantly increases the image size and slow down the devcontainer creation / update. To install them, manually run `pip install` after the devcontainer is launched:

```
pip install tensorflow
```

We'll add a devcontainer image with full dependencies including all the extra ML packages once the Remote - Containers extention supports [devcontainers/spec#6](https://github.com/devcontainers/spec/issues/6). [microsoft/vscode-remote-release#2067 (comment)](https://github.com/microsoft/vscode-remote-release/issues/2067#issuecomment-1075321633) indicates VS Code's devcontainer team is already working on implementing this feature. Once this feature is supported, we can have multiple `devcontainer.json` files with different configurations (for example, one for a smaller docker image with minimal dependencies and one for a larger docker image with full dependencies), and select one of them depending on development needs.

```json
// devcontainer.small.json
{
  "dockerComposeFile": "docker-compose.yml",
  "service": "mlflow-small",
  ...
}

// devcontainer.large.json
{
  "dockerComposeFile": "docker-compose.yml",
  "service": "mlflow-large",
  ...
}
```

## Development

See [DEVEOPMENT.md](./DEVELOPMENT.md).
