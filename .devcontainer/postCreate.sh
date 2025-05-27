#!/bin/bash

# Turn off git status check to improve zsh response speed: https://stackoverflow.com/a/25864063
git config --add oh-my-zsh.hide-status 1
git config --add oh-my-zsh.hide-dirty 1
git config --global --add safe.directory /workspaces/mlflow
pre-commit install --install-hooks
pip install --no-deps \
  -e . \
  -e ./dev/clint
