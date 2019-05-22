#!/usr/bin/env bash

set -ex
pushd ../mlflow/R/mlflow
Rscript -e 'install.packages("devtools", repos = "https://cloud.r-project.org")'
Rscript -e 'devtools::install_dev_deps(dependencies = TRUE)'
# Install Rd2md from source as a temporary fix for the rendering of code examples, until
# a release is published including the fixes in https://github.com/quantsch/Rd2md/issues/1
Rscript -e 'devtools::install_github("https://github.com/smurching/Rd2md", ref = "mlflow-patches")'
Rscript -e 'install.packages("rmarkdown", repos = "https://cloud.r-project.org")'
rm -rf man
Rscript -e "roxygen2::roxygenise()"
Rscript document.R
popd
