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

## Tracking

MLflow Tracking allows you to logging parameters, code versions,
metrics, and output files when running R code and for later visualizing
the results.

MLflow allows you to group runs under experiments, which can be useful
for comparing runs intended to tackle a particular task. You can create
and activate a new experiment locally using `mlflow` as follows:

``` r
library(mlflow)
mlflow_experiment("Test")
```

Then you can list your experiments directly from R,

``` r
mlflow_experiment_list()
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

## Projects

An MLflow Project is a format for packaging data science code in a
reusable and reproducible way.

MLflow projects are created, simply, by running your code under
`mlflow_run()` from R or the terminal:

``` bash
Rscript -e "mlflow::mlflow_run('train.R')"
```

With `train.R` performing training and logging as follows:

``` r
library(mlflow)

# read parameters
column <- mlflow_param("column", 1, "column index to use as feature")

# log total rows
mlflow_log("rows", nrow(iris))

# train model
model <- lm(Sepal.Width ~ iris[[column]], iris)

# log models intercept
mlflow_log("intercept", model$coefficients[["(Intercept)"]])
```

## Models

An MLflow Model is a standard format for packaging machine learning
models that can be used in a variety of downstream tools—for example,
real-time serving through a REST API or batch inference on Apache Spark.
They provide a convention to save a model in different “flavors” that
can be understood by different downstream tools.

To save a model use `mlflow_save_model()`. For instance, you can add the
following lines to the previous `train.R` script:

``` r
# train model (...)

# save model
mlflow_save_model(function(data) {
  predict(model, data)
})
```

And trigger a run with that will also save your model as follows:

``` bash
Rscript -e "mlflow::mlflow_run('train.R')"
```

Each MLflow Model is simply a directory containing arbitrary files,
together with an MLmodel file in the root of the directory that can
define multiple flavors that the model can be viewed in.

The directory containing the model looks as follows:

    ## [1] "MLmodel"     "r_model.bin"

and the model definition `mlflow-model/MLmodel` like:

    ## time_created: 1.5327199e+09
    ## flavors:
    ##   r_function:
    ##     version: 0.1.0
    ##     model: r_model.bin

## Deployment

MLflow provides tools for deployment on a local machine and several
production environments. You can use these tools to easily apply your
models in a production environment.

You can serve a model by running,

``` bash
mlflow rfunc serve mlflow-model
```

which is equivalent to
running,

``` bash
Rscript -e "mlflow_serve('mlflow-model')"
```

<img src="tools/readme/mlflow-serve-rfunc.png" class="screenshot" width=460 />
