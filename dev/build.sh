#!/usr/bin/env bash
set -e

SKINNY=0

for arg in "$@"
do
  if [ "$arg" == "--skinny" ]; then
    SKINNY=1
    break
  fi
done

if [ $SKINNY -eq 1 ]; then
  cat pyproject.skinny.toml > pyproject.toml
  echo "" >> README_SKINNY.rst
  cat README.rst >> README_SKINNY.rst
  cat README_SKINNY.rst > README.rst
  python -m build
  git restore .
else
  python -m build
fi
