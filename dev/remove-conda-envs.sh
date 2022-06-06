#!/usr/bin/env bash

set -ex

mlflow_envs=$(
  conda env list |                 # list (env name, env path) pairs
  cut -d' ' -f1 |                  # extract env names
  grep "^mlflow-[a-z0-9]\{40\}\$"  # filter envs created by mlflow
) || true

if [ ! -z "$mlflow_envs" ]; then
  for env in $mlflow_envs
  do
    conda remove --all --yes --name $env
  done
fi

conda clean --all --yes
conda env list

set +ex
