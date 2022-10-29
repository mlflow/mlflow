#!/usr/bin/env bash

set +exv

showHelp() {
cat << EOF
Usage: ./install-dev-env.sh [-d] [directory to install virtual environment] [-v] [-q] [-f] [-o] [override python version]
Development environment setup script for Python in linux-based Operating Systems (including OSX).
Note: this script will not work on Windows or MacOS M1 arm64 chipsets.

This script will:

  - Install pyenv if not installed
  - Retrieve the appropriate Python version (minimum required) for compatibility support
  - Check if the virtual environment already exists
    - If it does, prompt for replacement
      - If replacing, delete old virtual environment.
  - Create a virtual environment using the minimum required Python version based on previous step logic
  - Activate the environment
  - Install required dependencies for the dev environment

  Example usage:

  From root of MLflow repository on local with a destination virtualenv path of <MLFLOW_HOME>/.venvs/mlflow-dev:

  dev/dev-env-setup.sh -d $(pwd)/.venvs/mlflow-dev

  Note: it is recommended to preface virtualenv locations with a directory name prefaced by '.' (i.e., ".venvs").

  The default environment setup is for basic functionality, installing the minimum development requirements dependencies.
  To install the full development environment that supports working on all flavors and running all tests locally, set
  the flag '-f' or '--full'

-h,     --help        Display help

-d,     --directory   The path to install the virtual environment into

-f,     --full        Whether to install all dev requirements (Default: false)

-q,     --quiet       Whether to have pip install in quiet mode (Default: false)

-o,     --override    Override the python version

EOF
}

while :
do
  case "$1" in
    -d | --directory)
      directory="$2"
      shift 2
      ;;
    -f | --full)
      full="full"
      shift
      ;;
    -q | --quiet)
      quiet="quiet"
      shift
      ;;
    -o | --override)
      override_py_ver="$2"
      shift 2
      ;;
    -h | --help)
      showHelp
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "$verbose" ]]; then
  set -exv
fi

# Acquire the OS for this environment
case "$(uname -s)" in
  Darwin*)                       machine=mac;;
  Linux*)                        machine=linux;;
  *)                             machine=unknown;;
esac

quiet_command(){
  echo $( [[ -n $quiet ]] && printf %s '-q' )
}

minor_to_micro() {
  case $1 in
    "3.7") echo "3.7.14" ;;
    "3.8") echo "3.8.13" ;;
    "3.9") echo "3.9.13" ;;
    "3.10") echo "3.10.4" ;;
  esac
}

# Check if pyenv is installed and offer to install it if not present
pyenv_exist=$(command -v pyenv)

if [ -z "$pyenv_exist" ]; then
  if [ -z "$GITHUB_ACTIONS" ]; then
    read -p "pyenv is required to be installed to manage python versions. Would you like to install it? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
    echo
  fi
  if [[ $REPLY =~ ^[Yy]$ || -n "$GITHUB_ACTIONS" ]]; then
    if [[ "$machine" == mac ]]; then
      # Check if brew is installed and install it if it isn't present
      # Note: if xcode isn't installed, this will fail.
      if [ -z "$(command -v brew)" ]; then
        echo "Homebrew is required to install pyenv on MacOS. Installing in your home directory."
        bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      fi
      echo "Updating brew and installing pyenv..."
      echo "Note: this will probably take a considerable amount of time."
      brew update
      brew install pyenv
      brew install openssl readline sqlite3 xz zlib
    elif [[ "$machine" == linux ]]; then
      sudo apt-get update -y
      sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
      libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
      # Install pyenv from source
      git clone --depth 1 https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
      PYENV_ROOT="$HOME/.pyenv"
      PYENV_BIN="$PYENV_ROOT/bin"
      PATH="$PYENV_BIN:$PATH"
      if [ -n "$GITHUB_ACTIONS" ]; then
        echo "$PYENV_BIN" >> "$GITHUB_PATH"
        echo "PYENV_ROOT=$PYENV_ROOT" >> "$GITHUB_ENV"
      fi
    else
      echo "Unknown operating system environment: $machine exiting."
      exit 1
    fi
  else
    PYENV_README=https://github.com/pyenv/pyenv/blob/master/README.md
    echo "pyenv is required to use this environment setup script. Please install by following instructions here: $PYENV_README"
    exit 1
  fi
fi

MLFLOW_HOME=$(pwd)
rd="$MLFLOW_HOME/requirements"

# Get the minimum supported version for development purposes
min_py_version="3.8"

echo "The minimum version of Python to ensure backwards compatibility for MLflow development is: $(tput bold; tput setaf 3)$min_py_version$(tput sgr0)"

if [[ -n "$override_py_ver" ]]; then
  version_levels=$(grep -o '\.' <<< "$override_py_ver" | wc -l)
  if [[ $version_levels -eq 1 ]]; then
    PY_INSTALL_VERSION=$(minor_to_micro $override_py_ver)
  elif [[ $version_levels -eq 2 ]]; then
    PY_INSTALL_VERSION=$override_py_ver
  else
    echo "You must supply a python override version with either minor (e.g., '3.9') or micro (e.g., '3.9.5'). '$override_py_ver' is invalid."
    exit 1
  fi
else
  PY_INSTALL_VERSION=$(minor_to_micro $min_py_version)
fi

echo "The top-level dependencies that will be installed are: "

if [[ -n "$full" ]]; then
  files=("$rd/test-requirements.txt" "$rd/lint-requirements.txt" "$rd/doc-requirements.txt" "$rd/extra-ml-requirements.txt")
  echo "Files:"
  echo "MLflow test plugin: $MLFLOW_HOME/tests/resources/mlflow-test-plugin"
  echo "The local development branch of MLflow installed in editable mode with 'extras' requirements"
  echo "The following packages: "
else
  files=("$rd/test-requirements.txt" "$rd/lint-requirements.txt" "$rd/doc-requirements.txt")
fi
tail -n +1 "${files[@]}" | grep "^[^#= ]" | sort | cat

echo "$(tput setaf 2) Installing Python version $(tput bold)$PY_INSTALL_VERSION$(tput sgr0)"

# Install the Python version if it cannot be found
pyenv install -s "$PY_INSTALL_VERSION"
pyenv local "$PY_INSTALL_VERSION"
pyenv exec pip install $(quiet_command) --upgrade pip
pyenv exec pip install $(quiet_command) virtualenv

VENV_DIR="$directory/bin/activate"

# Check if the virtualenv already exists at the specified path
if [[ -d "$directory"  ]]; then
  if [ -z "$GITHUB_ACTIONS" ]; then
    read -p "A virtual environment is already located at $directory. Do you wish to replace it? $(tput bold; tput setaf 2)(y/n) $(tput sgr0)" -n 1 -r
    echo
  fi
  if [[ $REPLY =~ ^[Yy]$ || -n "$GITHUB_ACTIONS" ]]; then
    echo "Replacing Virtual environment in '$directory'. Installing new instance."
    pyenv exec virtualenv --clear "$directory"
  fi
else
  # Create a virtual environment with the specified Python version
  pyenv exec virtualenv --python "$PY_INSTALL_VERSION" "$directory"
fi

# Activate the virtual environment
# shellcheck disable=SC1090
source "$VENV_DIR"

echo "$(tput setaf 2)Current Python version: $(tput bold)$(python --version)$(tput sgr0)"
echo "$(tput setaf 3)Activated environment is located: $(tput bold) $directory/bin/activate$(tput sgr0)"

echo "Installing pip dependencies for development environment."

# Install current checked out version of MLflow (local)
pip install $(quiet_command) -e .[extras]

if [[ -n "$full" ]]; then
  # Install dev requirements and test plugin
  pip install $(quiet_command) -r "$MLFLOW_HOME/requirements/dev-requirements.txt"
  # Install test plugin
  pip install $(quiet_command) -e "$MLFLOW_HOME/tests/resources//mlflow-test-plugin"
  echo "Finished installing pip dependencies."
else
  files=("$rd/test-requirements.txt" "$rd/lint-requirements.txt" "$rd/doc-requirements.txt")
  for r in "${files[@]}";
  do
    pip install $(quiet_command) -r "$r"
  done
fi

echo "$(tput setaf 2; tput smul)Python packages that have been installed:$(tput rmul)"
echo "$(pip freeze)$(tput sgr0)"

command -v docker >/dev/null 2>&1 || echo "$(tput bold; tput setaf 1)A docker installation cannot be found. Please ensure that docker is installed to run all tests locally.$(tput sgr0)"

# Setup git environment configuration for proper signing of commits
git_user=$(git config user.name)
git_email=$(git config user.email)

if [[ -z "$git_email" || -z "$git_user" ]]; then
  read -p "Your git environment is not setup to automatically sign your commits. Would you like to configure it? $(tput bold; tput setaf 2)(y/n): $(tput sgr0)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter the user name you would like to have associated with your commit signature: " -r git_user_name
    echo
    git config --global user.name "$git_user_name"
    echo "Git user name set as: $(git config user.name)"
    read -p "Enter your email address for your commit signature: " -r git_user_email
    git config --global user.email "$git_user_email"
    echo "Git user email set as: $(git config user.email)"
  else
    echo "Failing to set git 'user.name' and 'user.email' will result in unsigned commits. Ensure that you sign commits manually for CI checks to pass."
  fi
fi

# setup pre-commit hooks
git config core.hooksPath "$MLFLOW_HOME/hooks"

echo "$(tput setaf 2)Your MLflow development environment can be activated by running: $(tput bold)source $VENV_DIR$(tput sgr0)"
