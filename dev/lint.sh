#!/usr/bin/env bash

# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR

echo -e "\n========== black ==========\n"
# Exclude proto files because they are auto-generated
black --check .

if [ $? -ne 0 ]; then
  echo '
To apply black foramtting, do one of the following:
- Run `pip install $(cat requirements/lint-requirements.txt | grep "^black==") && black .`
- Comment `autoformat` on the PR'
fi

echo -e "\n========== pylint ==========\n"
pylint $(git ls-files | grep '\.py$')

echo -e "\n========== rstcheck ==========\n"
rstcheck README.rst

if [[ "$err" != "0" ]]; then
  echo -e "\nOne of the previous steps failed, check above"
fi

test $err = 0
