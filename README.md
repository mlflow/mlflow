mlflow: R interface for MLflow
================

[![Build
Status](https://travis-ci.org/rstudio/mlflow.svg?branch=master)](https://travis-ci.org/rstudio/mlflow)
[![AppVeyor Build
Status](https://ci.appveyor.com/api/projects/status/github/rstudio/mlflow?branch=master&svg=true)](https://ci.appveyor.com/project/JavierLuraschi/mlflow)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/mlflow)](https://cran.r-project.org/package=mlflow)
[![codecov](https://codecov.io/gh/rstudio/mlflow/branch/master/graph/badge.svg)](https://codecov.io/gh/rstudio/mlflow)

  - Install [MLflow](https://mlflow.org/) from R to track experiments
    locally.
  - Connect to MLflow servers to share experiments with others.
  - Use MLflow to export models that can be served locally and remotely.

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

## Experiments

You can create a new experiment locally using `mlflow` as follows:

``` r
library(mlflow)
mlflow_experiments_create("Test")
```

    ## $experimentId
    ## [1] "1"

Then you can list your experiments directly from R,

``` r
mlflow_experiments()
```

    ##   experiment_id    name artifact_location
    ## 1             0 Default          mlruns/0
    ## 2             1    Test          mlruns/1

or using MLflows user interface by
running:

``` r
mlflow_ui()
```

<img src="tools/readme/mlflow-user-interface.png" class="screenshot" width=460 />

You can also use a MLflow server to track and share experiments, see
[running a tracking
server](https://www.mlflow.org/docs/latest/tracking.html#running-a-tracking-server),
and then make use of this server by running:

``` r
mlflow_tracking_url("http://tracking-server:5000")
```

Once the tracking url is defined, the experiments will be stored and
tracked in the specified server which others will also be able to
access.
