#!/usr/bin/env bash

set -ex
pushd ../mlflow/R/mlflow

# `gert` requires `libgit2`:
# https://github.com/r-lib/gert#installation
sudo add-apt-repository ppa:cran/libgit2
sudo apt-get update
sudo apt-get install --yes libssh2-1-dev libgit2-dev

Rscript -e 'install.packages("devtools", repos = "https://cloud.r-project.org")'
Rscript -e 'devtools::install_dev_deps(dependencies = TRUE)'
# Install Rd2md from source as a temporary fix for the rendering of code examples, until
# a release is published including the fixes in https://github.com/quantsch/Rd2md/issues/1
# Note that this commit is equivalent to commit 6b48255 of Rd2md master
# (https://github.com/quantsch/Rd2md/tree/6b4825579a2df8a22898316d93729384f92a756b)
# with a single extra commit to fix rendering of \link tags between methods in R documentation.
Rscript -e 'devtools::install_github("https://github.com/smurching/Rd2md", ref = "ac7b22bb")'
Rscript -e 'install.packages("rmarkdown", repos = "https://cloud.r-project.org")'
rm -rf man
Rscript -e "roxygen2::roxygenise()"
# remove mlflow-package doc temporarily because no rst doc should be generated for it.
rm man/mlflow-package.Rd
Rscript document.R
# roxygenize again to make sure the previously removed mlflow-packge doc is available as R helpfile
Rscript -e "roxygen2::roxygenise()"
popd
