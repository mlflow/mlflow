#!/usr/bin/env bash

MLFLOW_HOME="$(pwd)"
directory="$MLFLOW_HOME/.venvs/mlflow-dev"
REPO_ROOT=$(git rev-parse --show-toplevel)
rd="$REPO_ROOT/requirements"
VENV_DIR="$directory/bin/activate"
# Progress file to resume the script from where it exited previously
PROGRESS_FILE="$MLFLOW_HOME/dev-env-setup-progress"

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

  From root of MLflow repository on local with a destination virtualenv path of <REPO_ROOT>/.venvs/mlflow-dev:

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

-c,     --clean       Discard the previous installation progress and restart the setup from scratch

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
    -c | --clean)
      rm $PROGRESS_FILE
      shift
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

load_progress() {
  if [[ ! -f "$PROGRESS_FILE" ]]; then
    echo "0" > "$PROGRESS_FILE"
  fi
  cat "$PROGRESS_FILE"
}

PROGRESS=$(load_progress)

save_progress() {
  echo "$1" > "$PROGRESS_FILE"
  PROGRESS=$(load_progress)
}

quiet_command(){
  echo $( [[ -n $quiet ]] && printf %s '-q' )
}

minor_to_micro() {
  case $1 in
    "3.10") echo "3.10.13" ;;
  esac
}

# Check if brew is installed and install it if it isn't present
# Note: if xcode isn't installed, this will fail.
# $1: name of package that requires brew
check_and_install_brew() {
  # command -v returns exit code 1 if brew does not exist, which directly terminates our test script.
  # Appending `|| true` to ignore the exit code.
  if [ -z "$(command -v brew || true)" ]; then
    echo "Homebrew is required to install $1 on MacOS. Installing in your home directory."
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  echo "Updating brew..."
  brew update
}

# Compare two version numbers
# Usage: version_gt version1 version2
# Returns 0 (true) if version1 > version2, 1 (false) otherwise
version_gt() {
    IFS='.' read -ra VER1 <<< "$1"
    IFS='.' read -ra VER2 <<< "$2"

    # Compare each segment of the version numbers
    for (( i=0; i<"${#VER1[@]}"; i++ )); do
        # If VER2 is shorter and we haven't found a difference yet, VER1 is greater
        if [[ -z ${VER2[i]} ]]; then
            return 0
        fi

        # If some segments are not equal, return their comparison result
        if (( ${VER1[i]} > ${VER2[i]} )); then
            return 0
        elif (( ${VER1[i]} < ${VER2[i]} )); then
            return 1
        fi
    done

    # If all common length segments are same, the one with more segments is greater
    return $(( ${#VER1[@]} <= ${#VER2[@]} ))
}

# Check if pyenv is installed and offer to install it if not present
check_and_install_pyenv() {
  # command -v returns exit code 1 if pyenv does not exist, which directly terminates our test script.
  # Appending `|| true` to ignore the exit code.
  pyenv_exist=$(command -v pyenv || true)
  if [ -z "$pyenv_exist" ]; then
    if [ -z "$GITHUB_ACTIONS" ]; then
      read -p "pyenv is required to be installed to manage python versions. Would you like to install it? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
      echo
    fi
    if [[ $REPLY =~ ^[Yy]$ || -n "$GITHUB_ACTIONS" ]]; then
      if [[ "$machine" == mac ]]; then
        check_and_install_brew "pyenv"
        echo "Installing pyenv..."
        echo "Note: this will probably take a considerable amount of time."
        brew install pyenv
        brew install openssl readline sqlite3 xz zlib libomp
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
          echo "$PYENV_BIN" >>"$GITHUB_PATH"
          echo "PYENV_ROOT=$PYENV_ROOT" >>"$GITHUB_ENV"
        fi
      else
        echo "Unsupported operating system environment: $machine. This setup script only supports MacOS and Linux. For other operating systems, please follow the manual setup instruction here: https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#manual-python-development-environment-configuration "
        exit 1
      fi
    else
      PYENV_README=https://github.com/pyenv/pyenv/blob/master/README.md
      echo "pyenv is required to use this environment setup script. Please install by following instructions here: $PYENV_README"
      exit 1
    fi
  fi
}

check_and_install_min_py_version() {
  # Get the minimum supported version for development purposes
  min_py_version="3.10"

  echo "The minimum version of Python to ensure backwards compatibility for MLflow development is: $(
    tput bold
    tput setaf 3
  )$min_py_version$(tput sgr0)"

  if [[ -n "$override_py_ver" ]]; then
    version_levels=$(grep -o '\.' <<<"$override_py_ver" | wc -l)
    if [[ $version_levels -eq 1 ]]; then
      PY_INSTALL_VERSION=$(minor_to_micro $override_py_ver)
    elif [[ $version_levels -eq 2 ]]; then
      PY_INSTALL_VERSION=$override_py_ver
    else
      echo "You must supply a python override version with either minor (e.g., '3.10') or micro (e.g., '3.10.13'). '$override_py_ver' is invalid."
      exit 1
    fi
  else
    PY_INSTALL_VERSION=$(minor_to_micro $min_py_version)
  fi

  echo "$(tput setaf 2) Installing Python version $(tput bold)$PY_INSTALL_VERSION$(tput sgr0)"

  # Install the Python version if it cannot be found
  pyenv install -s "$PY_INSTALL_VERSION"
  pyenv local "$PY_INSTALL_VERSION"
  pyenv exec pip install $(quiet_command) --upgrade pip
  pyenv exec pip install $(quiet_command) virtualenv
}

# Check if the virtualenv already exists at the specified path
create_virtualenv() {
  if [[ -d "$directory" ]]; then
    if [ -z "$GITHUB_ACTIONS" ]; then
      read -p "A virtual environment is already located at $directory. Do you wish to replace it? $(
        tput bold
        tput setaf 2
      )(y/n) $(tput sgr0)" -n 1 -r
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
}

# Install mlflow dev version and required dependencies
install_mlflow_and_dependencies() {
  # Install current checked out version of mlflow (local)
  pip install -e .[extras]

  echo "Installing pip dependencies for development environment."
  if [[ -n "$full" ]]; then
    # Install dev requirements
    pip install -r "$rd/dev-requirements.txt"
    # Install test plugin
    pip install -e "$MLFLOW_HOME/tests/resources/mlflow-test-plugin"
  else
    files=("$rd/test-requirements.txt" "$rd/lint-requirements.txt" "$rd/doc-requirements.txt")
    for r in "${files[@]}"; do
      pip install -r "$r"
    done
  fi
  echo "Finished installing pip dependencies."

  echo "$(
    tput setaf 2
    tput smul
  )Python packages that have been installed:$(tput rmul)"
  echo "$(pip freeze)$(tput sgr0)"
}

check_docker() {
   command -v docker >/dev/null 2>&1 || echo "$(
    tput bold
    tput setaf 1
  )A docker installation cannot be found. Docker is optional but you may need it for developing some features and running all tests locally. $(tput sgr0)"
}

# Check if pandoc with required version is installed and offer to install it if not present
check_and_install_pandoc() {
  pandoc_version=$(pandoc --version | grep "pandoc" | awk '{print $2}')
  if [[ -z "$pandoc_version" ]] || ! version_gt "$pandoc_version" "2.2.1"; then
    if [ -z "$GITHUB_ACTIONS" ]; then
      read -p "Pandoc version 2.2.1 or above is an optional requirement for compiling documentation. Would you like to install it? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
      echo
    fi

    if [[ $REPLY =~ ^[Yy]$ || -n "$GITHUB_ACTIONS" ]]; then
      echo "Installing Pandoc..."
      if [[ "$machine" == mac ]]; then
        check_and_install_brew "pandoc"
        brew install pandoc
      elif [[ "$machine" == linux ]]; then
        # Install pandoc via deb package as `apt-get` gives too old version
        TEMP_DEB=$(mktemp) &&
          wget --directory-prefix $TEMP_DEB https://github.com/jgm/pandoc/releases/download/2.16.2/pandoc-2.16.2-1-amd64.deb &&
          sudo dpkg --install $(find $TEMP_DEB -name '*.deb') &&
          rm -rf $TEMP_DEB
      else
        echo "Unsupported operating system environment: $machine. This setup script only supports MacOS and Linux. For other operating systems, please follow the manual setup instruction here: https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#manual-python-development-environment-configuration "
        exit 1
      fi
    fi
  fi
}

# Set up pre-commit hooks and git environment configuration for proper signing of commits
set_pre_commit_and_git_signoff() {
  git_user=$(git config user.name)
  git_email=$(git config user.email)

  if [[ -z "$git_email" || -z "$git_user" ]]; then
    read -p "Your git environment is not setup to automatically sign your commits. Would you like to configure it? $(
      tput bold
      tput setaf 2
    )(y/n): $(tput sgr0)" -n 1 -r
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

  # Set up pre-commit hooks
  pre-commit install --install-hooks
}

# Execute mandatory setups with strict error handling
set +xv && set -e
# Mandatory setups
if [[ "$PROGRESS" -eq "0" ]]; then
  check_and_install_pyenv
  save_progress 1
fi
if [[ "$PROGRESS" -eq "1" ]]; then
  check_and_install_min_py_version
  save_progress 2
fi
if [[ "$PROGRESS" -eq "2" ]]; then
  create_virtualenv
  save_progress 3
fi
if [[ "$PROGRESS" -eq "3" ]]; then
  install_mlflow_and_dependencies
  save_progress 4
fi
if [[ "$PROGRESS" -eq "4" ]]; then
  set_pre_commit_and_git_signoff
  save_progress 5
fi
if [[ "$PROGRESS" -eq "5" ]]; then
  # Clear progress file if all mandatory steps are executed successfully
  rm $PROGRESS_FILE
fi

# Execute optional setups without strict error handling
set +exv
# Optional setups
check_and_install_pandoc
check_docker

echo "$(tput setaf 2)Your MLflow development environment can be activated by running: $(tput bold)source $VENV_DIR$(tput sgr0)"
