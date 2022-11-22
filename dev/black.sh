#!/usr/bin/env bash
err=0
trap 'err=1' ERR

if [ -z "${GITHUB_ACTIONS}" ]; then
  black --check . "$@"
else
  black --check "$@" > .black-output 2>&1
  sed 's/^would reformat \(.*\)/\1: This file is unformatted. Run `black .` or comment `@mlflow-automation autoformat` on the PR if you'\''re an MLflow maintainer./' .black-output
fi

test $err = 0
