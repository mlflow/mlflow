#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

pytest tests --quiet --requires-ssh --ignore-flavors --ignore=tests/examples --ignore=tests/recipes

LIGHTNING_VERSION=`python -c "import pytorch_lightning as pl; print(pl.__version__)"`
pip uninstall -y pytorch_lightning
pytest tests/pytorch/test_tensorboard_autolog.py
pip install pytorch_lightning==$LIGHTNING_VERSION

test $err = 0
