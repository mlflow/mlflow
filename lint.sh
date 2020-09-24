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

# Check Python code for style issues using black
./dev/run-black-linter.sh

# pylint's `--ignore` option filters files based on their base names, not paths.
# see: http://pylint.pycqa.org/en/latest/user_guide/run.html#command-line-options
# This behavior might cause us to unintentionally ignore some files.
# To avoid this issue, select files to lint using `git ls-files` and `grep`.
# Another advantage of this approach is we can apply pylint to all python scripts
# without creating `__init__.py` in all directories.
exclude="^\($(join "\|" "${exclude_dirs[@]}")\)/.\+\.py$"
include="^\($(join "\|" "${include_dirs[@]}")\)/.\+\.py$"
msg_template="{path} ({line},{column}): [{msg_id} {symbol}] {msg}"

git ls-files | grep $include | grep -v $exclude | \
xargs pylint --msg-template="$msg_template" --rcfile="$FWDIR/pylintrc"

rstcheck README.rst
