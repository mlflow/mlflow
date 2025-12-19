#!/usr/bin/env bash
# Install from master:
# curl -LsSf https://raw.githubusercontent.com/mlflow/mlflow/HEAD/dev/install-skinny.sh | sh
#
# Install from a specific branch:
# curl -LsSf https://raw.githubusercontent.com/mlflow/mlflow/HEAD/dev/install-skinny.sh | sh -s <branch>
#
# Install from a specific PR:
# curl -LsSf https://raw.githubusercontent.com/mlflow/mlflow/HEAD/dev/install-skinny.sh | sh -s pull/<pr_num>/merge
REF=${1:-HEAD}

# Fetching the entire repo is slow. Use sparse-checkout to only fetch the necessary files.
TEMP_DIR=$(mktemp -d)
git clone --filter=blob:none --no-checkout https://github.com/mlflow/mlflow.git $TEMP_DIR
cd $TEMP_DIR
# Exclude the mlflow/server/js folder as it contains frontend JavaScript files not needed for mlflow-skinny installation.
git sparse-checkout set --no-cone /mlflow /libs/skinny /pyproject.toml '!/mlflow/server/js/*'
git fetch origin "$REF"
git config advice.detachedHead false
git checkout FETCH_HEAD
OPTIONS=$(if pip freeze | grep -q "mlflow-skinny @"; then echo "--force-reinstall --no-deps"; fi)
pip install $OPTIONS ./libs/skinny
rm -rf $TEMP_DIR
