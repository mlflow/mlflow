#!/usr/bin/env bash

set -ex
pushd ../mlflow/R/mlflow
Rscript -e 'install.packages("devtools", repos = "https://cloud.r-project.org")'
Rscript -e 'devtools::install_dev_deps(dependencies = TRUE)'
rm -rf man
Rscript -e "roxygen2::roxygenise()"
Rscript document.R
popd
