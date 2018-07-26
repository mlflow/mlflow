mlflow: R interface for MLflow
================

[![Build
Status](https://travis-ci.org/rstudio/mlflow.svg?branch=master)](https://travis-ci.org/rstudio/mlflow)
[![AppVeyor Build
Status](https://ci.appveyor.com/api/projects/status/github/rstudio/mlflow?branch=master&svg=true)](https://ci.appveyor.com/project/JavierLuraschi/mlflow)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/mlflow)](https://cran.r-project.org/package=mlflow)
[![codecov](https://codecov.io/gh/rstudio/mlflow/branch/master/graph/badge.svg)](https://codecov.io/gh/rstudio/mlflow)

  - Install [MLflow]() from R to manage models and experiments locally.
  - Connect to remote MLflow servers to share, deploy and server models.

## Installation

You can install **mlflow** from GitHub as follows:

``` r
devtools::install_github("rstudio/mlflow")
```

Then, install MLflow to manage models and experiments locally:

``` r
library(mlflow)
mlflow_install()
```

To upgrade to the latest version of mlflow, run the following command
and restart your r session:

``` r
devtools::install_github("rstudio/mlflow")
```

## Connecting

You can connect to both local instances of MLflow as well as remote
MLflow servers. Here we’ll connect to a local instance of MLflow via the
`mlflow_connect()` function:

``` r
library(mlflow)
mc <- mlflow_connect()
```

You can then launch the MLflow user interface by
running:

``` r
mlflow_ui(mc)
```

<img src="tools/readme/mlflow-user-interface.png" class="screenshot" width=460 />

## Experiments

To list experiments,
    run:

``` r
mlflow_experiments(mc)
```

    ##   experiment_id    name                             artifact_location
    ## 1             0 Default /Users/javierluraschi/RStudio/mlflow/mlruns/0

## Termination

To terminate a local MLflow instance, restart your R session or run
explicitly:

``` r
mlflow_disconnect(mc)
```
