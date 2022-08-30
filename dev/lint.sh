#!/usr/bin/env bash

# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR

echo -e "\n========== black ==========\n"
if [ -z "${GITHUB_ACTIONS}" ]; then
  black --check .
else
  black --check . > .black-output 2>&1
  sed 's/^would reformat \(.*\)/\1: This file is unformatted. Run `black .` or comment `@mlflow-automation autoformat` on the PR to format./' .black-output
fi

echo -e "\n========== pylint ==========\n"
pylint $(git ls-files | grep '\.py$')

if [[ -f "README.rst" ]]; then
  echo -e "\n========== rstcheck ==========\n"
  rstcheck README.rst
fi

if [[ "$err" != "0" ]]; then
  echo -e "\nOne of the previous steps failed, check above"
fi

test $err = 0
