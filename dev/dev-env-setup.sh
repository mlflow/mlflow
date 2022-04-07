#!/usr/bin/env bash

set +exv

showHelp() {
cat << EOF
Usage: ./install-dev-env.sh [-n] [environment name] [-d] [directory to install virtual environment] [-v]
Development environment setup script for Python.
This script will:

  - Install pyenv if not installed
  - Retrieve the appropriate Python version (minimum required) for compatibility support
  - Create a virtual environment using the minimum required Python version
  - Activate the environment
  - Install required dependencies for the dev envrionment

-h, -help,        --help        Display help

-n, -name,        --name        The name of the conda environment that you would like to create.

-d, -directory    --directory   The path to install the virtual environment into

-v, -verbose      --verbose     Whether to fill stdout with every command or not

EOF
}

tput_r="tput setaf 1"

export verbose=0
while getopts "n:d:vh" opt
do
  case "$opt" in
    n) name="$OPTARG" ;;
    d) directory="$OPTARG" ;;
    v) verbose=1 ;;
    h) showHelp; exit ;;
    *) showHelp; exit ;;
  esac
done

if [[ $verbose = 1 ]]; then
  set -exv
fi

# Check if pyenv is installed and offer to install it if not present
pyenv_exist=$(command -v pyenv)

if [ -z "$pyenv_exist" ]; then
  read -p "pyenv is required to be installed to manage python versions. Would you like to install it? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Updating brew and installing pyenv..."
    echo "Note: this will probably take a considerable amount of time."
    brew update
    brew install pyenv
  else
    exit 1
  fi
fi

MLFLOW_HOME=$(pwd)

# Get the minimum supported version from MLflow to ensure any feature development adheres to legacy Python versions
min_py_version=$(grep "python_requires=" "$MLFLOW_HOME/setup.py" | grep -E -o "([0-9]{1,}\.)+[0-9]{1,}")

# Resolve a minor version to the latest micro version
case $min_py_version in
  "3.7") PY_INSTALL_VERSION="3.7.13" ;;
  "3.8") PY_INSTALL_VERSION="3.8.13" ;;
  "3.9") PY_INSTALL_VERSION="3.9.11" ;;
  "3.10") PY_INSTALL_VERSION="3.10.3" ;;
esac

# Install the Python version if it cannot be found
pyenv install -s $PY_INSTALL_VERSION
pyenv local $PY_INSTALL_VERSION
pyenv exec pip install --upgrade pip
pyenv exec pip install virtualenv

# Create a virtual environment with the specified Python version
pyenv exec virtualenv "$directory"
source $directory/bin/activate

echo "Current Python version: $(python --version)"




# check if the python version is already installed?

# check if virtualenv already exists and prompt for replacement from scratch
# deactivate the environment if active
# rm -rf path/root/of/venv

# create new virtualenv

# activate the environment (need a path for the environment supplied by the user)
# source path/to/env/bin/activate

#
#
## validate conda
#command -v conda >/dev/null 2>&1 || { echo >&2 "Conda must be installed in order to use this script."; exit 1; }
#
#existing_envs=$(conda env list | grep "\b$name\b\s")
#
#if [ -n "$existing_envs" ]; then
#  read -p "$($tput_r)An existing environment was found with the name '$name'. Do you wish to replace this environment? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
#  echo
#  if [[ $REPLY =~ ^[Yy]$ ]]; then
#    echo "removing environment!" #replace with conda remove
#  else
#    exit 1
#  fi
#fi

#conda create --name $name --python=$python_version
#
#conda activate $name

# install the base requirements

# install the test module

# install







