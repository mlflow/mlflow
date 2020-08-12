#!/usr/bin/env bash

set -e

FWDIR="$(cd "`dirname $0`"; pwd)"
cd "$FWDIR"

# https://stackoverflow.com/a/17841619
function join {
  local d=$1
  shift
  echo -n "$1"
  shift
  printf "%s" "${@/#/$d}"
}

include_dirs=(
  "mlflow"
  "tests"
)

exclude_dirs=(
  "mlflow/protos"
  "mlflow/server/js"
  "mlflow/store/db_migrations"
  "mlflow/temporary_db_migrations_for_pre_1_users"
)

# Exclude proto files because they are auto-generated
black --check --line-length=100 --exclude=mlflow/protos .

exclude=$(join "," "${exclude_dirs[@]}")
include=$(join " " "${include_dirs[@]}")
pycodestyle --max-line-length=100 --ignore=E203,W503 --exclude=$exclude -- $include

# pylint's `--ignore` option filters files based on their base names, not paths.
# see: http://pylint.pycqa.org/en/latest/user_guide/run.html#command-line-options
# This behavior might cause us to unintentionally ignore some files.
# To avoid this issue, select files to lint using `git ls-files` and `grep`.
# This approach also solves another issue where pylint ignores directories that
# don't contain `__init__.py`.
exclude="^\($(join "\|" "${exclude_dirs[@]}")\)/.\+\.py$"
include="^\($(join "\|" "${include_dirs[@]}")\)/.\+\.py$"
msg_template="{path} ({line},{column}): [{msg_id} {symbol}] {msg}"

git ls-files | grep $include | grep -v $exclude | \
xargs pylint --msg-template="$msg_template" --rcfile="$FWDIR/pylintrc"

rstcheck README.rst
