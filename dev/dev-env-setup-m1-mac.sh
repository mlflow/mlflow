#!/bin/bash

display_help() {
  cat << EOF

  This script installs a development environment designed to support M1 arm64 apple silicon
  computers. It utilizes miniforge to build a minimal conda (or mamba) environment that only
  references packages within conda-forge.

  Arguments:
    -e -env --environment
      This is the name of the conda environment that will be built. If an existing environment
      with this name exists, a prompt will ask if you would like to overwrite the existing
      environment.
      WARNING: Overwriting the miniconda environment will remove all libraries in the preexisting
      environment. Ensure you are certain that you want to overwrite an existing environment!
    -q -quiet --quiet
      An optional flag to suppress statements to stdout during package installation.
      Note that certain conda install statements will still print.
    -f -force --force
      A flag to accept all prompts (selecting 'yes' for all installation and overwrites)
    -r -remove --remove
      An option that will remove an environment that has been specified with the '-e' argument.

  Usage (from MLflow repository root):
    dev/dev-env-setup-m1-mac.sh -e "mlflow-dev" -q -f

EOF
}

while :
do
  case "$1" in
    -e | -env | --environment)
      environment="$2"
      shift 2
      ;;
    -h | -help | --help)
      display_help
      exit 0
      ;;
    -q | -quiet | --quiet)
      quiet="quiet"
      shift
      ;;
    -f | -force | --force)
      force="force"
      shift
      ;;
    -r | -remove | --remove)
      remove="remove"
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

if [[ -z $environment ]]; then
  echo "An environment name must be specified with the '-e' '-env' or '--environment' arguments. Please run this script with the '-h' argument for more details."
  exit 1
fi

function quietcommand(){
  echo $( [[ -n $quiet ]] && printf %s '-q' )
}

function autoaccept(){
  echo $( [[ -n $force ]] && printf %s '-y')
}

# Remove a named conda environment if it exists
if [[ -n $remove ]]; then
  if conda env list | grep -q "$environment"; then
    eval "$(conda shell.bash hook)"
    conda activate --no-stack "$environment"
    echo "$(tput setaf 3; tput smul)The following packages will be removed along with the environment:$(tput rmul)"
    echo "$(python -m pip freeze)$(tput sgr0)"
    if [ -z $force ]; then
      read -p "$(tput setaf 1; tput bold)Do you wish to remove this environment? Y/N $(tput sgr0)"
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing environment '$environment'."
        conda deactivate
        conda env remove --name $environment
        echo "Environment $environment removed."
        exit 0
      else
        echo "Aborting environment removal."
        exit 0
      fi
    else
      echo "Removing environment '$environment'."
      conda deactivate
      conda env remove --name $environment
      echo "Environment $environment removed."
      exit 0
    fi
  else
    echo "The environment '$environment' is not a registered conda environment."
    echo "The environments present on this system are: "
    conda env list
    exit 0
  fi
fi


MLFLOW_HOME=$(pwd)
rd="$MLFLOW_HOME/requirements"

# Check if a conda manager is installed.
# If none found, install miniforge.
if [ -z "$(command -v conda)" ]; then

  echo "A conda environment manager has not been found. Downloading miniforge..."

  bash -c "$(curl -fsSLo /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-"$(uname -m)".sh)"
  bash /tmp/Miniforge3.sh -b

fi

conda update -n base $(autoaccept) $(quietcommand) -c defaults conda

# Check for the minimally supported version on conda-forge for python (likely a higher version than what MLflow supports)
py_min_ver="$(conda search python --json | jq -r '.python[0].version' | grep -o "^[0-9]*\.[0-9]*")"

# Check if an environment already exists with the name specified.
if conda env list | grep -q "$environment"; then
  if [ -z $force ]; then
    read -p "A conda environment already exists with the name $environment. Do you want to replace it? $(tput bold; tput setaf 3)y/n$(tput sgr0)" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      conda create -y $(quietcommand) -n "$environment" python="$py_min_ver"
    else
      echo "Conda environment $environment is untouched. Create a new environment with the '-e' argument to use this script if you don't want to override the environment."
      exit 1
    fi
  else
    conda create $(autoaccept) $(quietcommand) -n "$environment" python="$py_min_ver"
  fi
else
  conda create $(autoaccept) $(quietcommand) -n "$environment" python="$py_min_ver"
fi

# Activate the environment locally in this subprocess so we can install packages
eval "$(conda shell.bash hook)"

conda activate --no-stack "$environment"

# Install packages for dev environment

# Setup for Tensorflow on M1
conda install $(quietcommand) $(autoaccept) -c apple tensorflow-deps
python -m pip install $(quietcommand) tensorflow-macos
python -m pip install $(quietcommand) tensorflow-metal
if [ -n "$force" ]; then
  yes | brew install $(quietcommand) libjpeg
else
  brew install $(quietcommand) libjpeg
fi
conda install $(quietcommand) $(autoaccept) -c conda-forge matplotlib

# Setup for ONNX and LightGBM on M1
conda install $(autoaccept) $(quietcommand) -c conda-forge onnx onnxruntime lightgbm

# Setup for paddlepaddle on M1
conda install $(autoaccept) $(quietcommand) -c paddle paddlepaddle

# Install the prerequisites for prophet dependencies to build properly
python -m pip install $(quietcommand) Cython numpy

# Install base requirements for prophet to build
tmp_dir=$(mktemp -d)
python -m pip download $(quietcommand) --no-deps --dest "$tmp_dir" --no-cache-dir prophet
tar -zxvf "$tmp_dir"/*.tar.gz -C "$tmp_dir"
python -m pip install $(quietcommand) -r "$(find "$tmp_dir" -name requirements.txt)"
rm -rf "$tmp_dir"
python -m pip install $(quietcommand) prophet

# Install core requirements for MLflow development
python -m pip install $(quietcommand) -r "$rd/small-requirements.txt"
python -m pip install $(quietcommand) -r "$rd/large-requirements.txt"
python -m pip install $(quietcommand) -r "$rd/lint-requirements.txt"
python -m pip install $(quietcommand) -r "$rd/doc-requirements.txt"

# Install the ML packages that have the ability to be installed from pypi for M1 chips
tmp_dir=$(mktemp -d)
grep -v '#\|onnx\|paddle\|azure\|lightgbm\|prophet\|tensorflow\|mxnet' ./requirements/extra-ml-requirements.txt > $tmp_dir/requirements.txt
python -m pip install $(quietcommand) -r "$(find "$tmp_dir" -name requirements.txt)"
rm -rf "$tmp_dir"

# Install a development version of MLflow from the local branch
python -m pip install $(quietcommand) -e .[extras]
python -m pip install $(quietcommand) -e "$MLFLOW_HOME/tests/resources//mlflow-test-plugin"

# To prevent a liblapack dylib import error, force reinstall numpy, scipy, and scikit-learn
conda install --force-reinstall $(quietcommand) numpy scikit-learn scipy

echo "$(tput setaf 2; tput smul)Python packages that have been installed:$(tput rmul)"
echo "$(python -m pip freeze)$(tput sgr0)"

echo "Environment is configured for use. Activate by running $(tput bold; tput setaf 2)'conda activate $environment'$(tput sgr0)."
exit 0