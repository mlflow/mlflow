#!/bin/bash

# Because $OSTYPE could be cygwin, msys, or win32, etc on Windows. The possibilities are endless.
if [[ "${OSTYPE}" != linux* && "${OSTYPE}" != darwin* ]]; then
  export PATH=\
"${PATH}:\
$(
  R_HOME='/c/R'
  MINICONDA_ENVS_DIR='/c/Miniconda/envs'
  MLFLOW_CONDA_ENV_NAME="$(${R_HOME}/bin/Rscript -e 'cat(mlflow:::mlflow_conda_env_name())')"
  ADDITIONAL_PATHS_ARR=(
    "${R_HOME}/bin"
    "${MINICONDA_ENVS_DIR}"/{'test-environment',"${MLFLOW_CONDA_ENV_NAME}"}/'Scripts'
  )

  export IFS=':'
  echo "${ADDITIONAL_PATHS_ARR[*]}"
)
"

  # fail fast if any of these required binaries is not present
  which waitress-serve
  which mlflow
fi
