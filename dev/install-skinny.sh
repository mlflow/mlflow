#!/usr/bin/env bash
PR_NUMBER=$1

# Fetching the entire repo is slow. Use sparse-checkout to only fetch the necessary files.
TEMP_DIR=$(mktemp -d)
git clone --filter=blob:none --no-checkout https://github.com/mlflow/mlflow.git $TEMP_DIR
cd $TEMP_DIR
git sparse-checkout set --no-cone /mlflow /skinny /pyproject.toml
git fetch origin pull/$PR_NUMBER/merge
git config advice.detachedHead false
git checkout FETCH_HEAD
OPTIONS=$(if pip freeze | grep -q "mlflow-skinny @"; then echo "--force-reinstall --no-deps"; fi)
pip install --no-build-isolation $OPTIONS ./skinny
rm -rf $TEMP_DIR
