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
      WARNING: this will remove all libraries in the environment. Ensure you are certain that you
      want to overwrite an existing environment!
    -q -quiet --quiet
      An optional flag to suppress pip statements to stdout during install.
    -v -verbose --verbose
      A flag to silence conda stdout statements.

  Usage (from MLflow repository root):
    dev/dev-env-setup-miniforge.sh -d "~/miniforge/mlflow-dev" -q

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
    -v | -verbose | --verbose)
      verbose="verbose"
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

function quietpip(){
  echo $( [[ -n $quiet ]] && printf %s '-q' )
}

function quietconda(){
  echo $( [[ -n $verbose ]] && printf %s '-q')
}

MLFLOW_HOME=$(pwd)
rd="$MLFLOW_HOME/requirements"

# Check if a conda manager is installed.
# If none found, install miniforge.
if [ -z "$(command -v conda)" ]; then

  echo "A conda environment manager has not been found. Downloading miniforge..."

  bash -c "$(curl -fsSLo /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-"$(uname -m)".sh)"
  bash /tmp/Miniforge3.sh -b

fi

conda update -n base -y $(quietconda) -c defaults conda

# Check for the minimally supported version on conda-forge for python (likely a higher version than what MLflow supports)
py_min_ver="$(conda search python --json | jq -r '.python[0].version' | grep -o "^[0-9]*\.[0-9]*")"

# Check if an environment already exists with the name specified.
if conda env list | grep -q "$environment"; then
  read -p "A conda environment already exists with the name $environment. Do you want to replace it? $(tput bold; tput setaf 3)y/n$(tput sgr0)" -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda create -y $(quietconda) -n "$environment" python="$py_min_ver"
  else
    echo "Conda environment $environment is untouched. Create a new environment with the '-e' argument to use this script if you don't want to override the environment."
    exit 1
  fi
else
  conda create -y $(quietconda) -n "$environment" python="$py_min_ver"
fi

# Activate the environment locally in this subprocess so we can install packages
eval "$(conda shell.bash hook)"

conda activate --no-stack "$environment"

# Install packages for dev environment

# TODO: clean up these operations with functions

# Setup for Tensorflow on M1
conda install $(quietconda) -y -c apple tensorflow-deps
python -m pip install $(quietpip) tensorflow-macos
python -m pip install $(quietpip) tensorflow-metal
yes | brew install $(quietconda) libjpeg
conda install $(quietconda) -y -c conda-forge matplotlib

# Setup for ONNX and LightGBM
conda install -y $(quietconda) -c conda-forge onnx onnxruntime lightgbm

# Setup for paddlepaddle
conda install $(quietconda) -c paddle paddlepaddle

python -m pip install $(quietpip) Cython numpy

# Install base requirements for prophet to build
tmp_dir=$(mktemp -d)
python -m pip download $(quietpip) --no-deps --dest "$tmp_dir" --no-cache-dir prophet
tar -zxvf "$tmp_dir"/*.tar.gz -C "$tmp_dir"
python -m pip install $(quietpip) -r "$(find "$tmp_dir" -name requirements.txt)"
rm -rf "$tmp_dir"
python -m pip install $(quietpip) prophet

python -m pip install $(quietpip) -r "$rd/small-requirements.txt"
python -m pip install $(quietpip) -r "$rd/large-requirements.txt"
python -m pip install $(quietpip) -r "$rd/lint-requirements.txt"
python -m pip install $(quietpip) -r "$rd/doc-requirements.txt"

python -m pip install $(quietpip) keras scikit-learn mxnet fastai spacy torch torchvision pytorch_lightning xgboost catboost statsmodels h2o pyspark shap pmdarima diviner

python -m pip install $(quietpip) -e .[extras]
python -m pip install $(quietpip) -e "$MLFLOW_HOME/tests/resources//mlflow-test-plugin"




echo "$(tput setaf 2; tput smul)Python packages that have been installed:$(tput rmul)"
echo "$(python -m pip freeze)$(tput sgr0)"

echo "Environment is configured for use. Activate by running $(tput bold; tput setaf 2)'conda activate $environment'$(tput sgr0)."