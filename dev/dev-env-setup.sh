#!/usr/bin/env bash

set +exv

showHelp() {
cat << EOF
Usage: ./install-dev-env.sh [-n] [-pv]
Development environment setup script for Python.
This script will generate a a new conda environment, activate it, and install all required dev dependencies.

NOTE: you will need either conda or miniconda to be installed in your environment to run this script.
miniconda can be downloaded from here: https://docs.conda.io/en/latest/miniconda.html

-h, -help,        --help             Display help

-n, -name,        --name             The name of the conda environment that you would like to create.

-p, -python       --python-version   The version of Python to install for the conda environment.

-v, -verbose      --verbose          Whether to fill stdout with every command or not

EOF
}

tput_r="tput setaf 1"

export verbose=0
while getopts "n:p:vh" opt
do
  case "$opt" in
    n) name="$OPTARG" ;;
    p) python_version="$OPTARG" ;;
    v) verbose=1 ;;
    h) showHelp; exit ;;
    *) showHelp; exit ;;
  esac
done

if [[ $verbose = 1 ]]; then
  set -exv
fi

MLFLOW_HOME=$(pwd)

# validate conda
command -v conda >/dev/null 2>&1 || { echo >&2 "Conda must be installed in order to use this script."; exit 1; }

existing_envs=$(conda env list | grep "\b$name\b\s")

if [ -n "$existing_envs" ]; then
  read -p "$($tput_r)An existing environment was found with the name '$name'. Do you wish to replace this environment? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "removing environment!" #replace with conda remove 
  else
    exit 1
  fi
fi

#conda create --name $name --python=$python_version
#
#conda activate $name

# install the base requirements

# install the test module

# install

echo "name is: $name"
echo "python version to be installed: $python_version"






