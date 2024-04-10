#!/usr/bin/env bash
set -e

SKINNY=0
PYTHON_PATH=python

for arg in "$@"
do
  if [ "$arg" == "--skinny" ]; then
    SKINNY=1
  elif [[ "$arg" == --python-path=* ]]; then
    PYTHON_PATH="${arg#*=}"
  fi
done

if [ $SKINNY -eq 1 ]; then
  cat pyproject.skinny.toml > pyproject.toml
  echo "" >> README_SKINNY.rst
  cat README.rst >> README_SKINNY.rst
  cat README_SKINNY.rst > README.rst
  $PYTHON_PATH -m build
  git restore .
else
  $PYTHON_PATH -m build
fi
