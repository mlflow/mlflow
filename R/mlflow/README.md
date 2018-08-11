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
mlflow_list_experiments()
```

    ##   experiment_id    name artifact_location
    ## 1             0 Default          mlruns/0
    ## 2             1    Test          mlruns/1

or using MLflows user interface by
running:

``` r
mlflow_ui()
```

<img src="tools/readme/mlflow-user-interface.png" class="screenshot" width=520 />

You can also use a MLflow server to track and share experiments, see
[running a tracking
server](https://www.mlflow.org/docs/latest/tracking.html#running-a-tracking-server),
and then make use of this server by running:

``` r
mlflow_set_tracking_uri("http://tracking-server:5000")
```

Once the tracking url is defined, the experiments will be stored and
tracked in the specified server which others will also be able to
access.

## Projects

An MLflow Project is a format for packaging data science code in a
reusable and reproducible way.

MLflow projects can be [explicitly
created](https://www.mlflow.org/docs/latest/projects.html#specifying-projects)
or implicitly used by running `R` with `mlflow` from the terminal as
follows:

``` bash
mlflow run examples/R/simple --entry-point train.R
```

Notice that is equivalent to running from `examples/R/simple`,

``` bash
Rscript -e "mlflow::mlflow_source('train.R')"
```

and `train.R` performing training and logging as follows:

``` r
library(mlflow)

# read parameters
column <- mlflow_log_param("column", 1)

# log total rows
mlflow_log_metric("rows", nrow(iris))

# train model
model <- lm(Sepal.Width ~ iris[[column]], iris)

# log models intercept
mlflow_log_metric("intercept", model$coefficients[["(Intercept)"]])
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
mlflow_save_model(function(df, model) {
  predict(model, df)
})
```

And trigger a run with that will also save your model as follows:

``` bash
mlflow run train.R
```

Each MLflow Model is simply a directory containing arbitrary files,
together with an MLmodel file in the root of the directory that can
define multiple flavors that the model can be viewed in.

The directory containing the model looks as follows:

    ## [1] "MLmodel"     "r_model.bin"

and the model definition `model/MLmodel` like:

    ## time_created: 1.5339457e+09
    ## flavors:
    ##   r_function:
    ##     version: 0.1.0
    ##     model: r_model.bin

Later on, the model can be deployed which will perform predictions using
`mlflow_predict()`:

``` r
mlflow_predict("model", iris)
```

    ## 3.103334366486683.115711326079513.128088285672343.134276765468753.10952284628313.084768927097443.134276765468753.10952284628313.146653725061583.115711326079513.084768927097443.121899805875933.121899805875933.1528422048583.060015007911783.06620348770823.084768927097443.103334366486683.06620348770823.103334366486683.084768927097443.103334366486683.134276765468753.103334366486683.121899805875933.10952284628313.10952284628313.097145886690273.097145886690273.128088285672343.121899805875933.084768927097443.097145886690273.078580447301023.115711326079513.10952284628313.078580447301023.115711326079513.146653725061583.103334366486683.10952284628313.140465245265173.146653725061583.10952284628313.103334366486683.121899805875933.103334366486683.134276765468753.090957406893853.10952284628312.985753250354813.02288412913332.991941730151223.078580447301023.016695649336883.06620348770823.029072608929713.115711326079513.010507169540473.097145886690273.10952284628313.053826528115373.047638048318953.041449568522543.072391967504613.004318689744053.072391967504613.060015007911783.035261088726123.072391967504613.053826528115373.041449568522543.029072608929713.041449568522543.02288412913333.010507169540472.998130209947643.004318689744053.047638048318953.06620348770823.078580447301023.078580447301023.060015007911783.047638048318953.084768927097443.047638048318953.004318689744053.029072608929713.072391967504613.078580447301023.078580447301023.041449568522543.060015007911783.10952284628313.072391967504613.06620348770823.06620348770823.035261088726123.103334366486683.06620348770823.029072608929713.060015007911782.979564770558393.029072608929713.016695649336882.948622371576323.115711326079512.967187810965573.004318689744052.973376290761983.016695649336883.02288412913332.998130209947643.06620348770823.060015007911783.02288412913333.016695649336882.942433891779912.942433891779913.047638048318952.991941730151223.072391967504612.942433891779913.029072608929713.004318689744052.973376290761983.035261088726123.041449568522543.02288412913332.973376290761982.960999331169152.930056932187083.02288412913333.029072608929713.041449568522542.942433891779913.029072608929713.02288412913333.047638048318952.991941730151223.004318689744052.991941730151223.060015007911782.998130209947643.004318689744053.004318689744053.029072608929713.016695649336883.035261088726123.05382652811537

## Deployment

MLflow provides tools for deployment on a local machine and several
production environments. You can use these tools to easily apply your
models in a production environment.

You can serve a model by running,

``` bash
mlflow rfunc serve model
```

which is equivalent to
running,

``` bash
Rscript -e "mlflow_serve('model')"
```

<img src="tools/readme/mlflow-serve-rfunc.png" class="screenshot" width=520 />

You can also run:

``` bash
mlflow rfunc predict model data.json
```

which is equivalent to running,

``` bash
Rscript -e "mlflow_predict('model', 'data.json')"
```

## Dependencies

When running a project, `mlflow_snapshot()` is automatically called to
generate a `r-dependencies.txt` file which contains a list of required
packages and versions.

However, restoring dependencies is not automatic since it’s usually an
expensive operation. To restore dependencies run:

``` r
mlflow_restore()
```

Notice that the `MLFLOW_SNAPSHOT_CACHE` environment variable can be set
to a cache directory to improve the time required to restore
dependencies.

## RStudio

To enable fast iteration while tracking with MLflow improvements over a
model, [RStudio 1.2.897](https://dailies.rstudio.com/) an be configured
to automatically trigger `mlflow_run()` when sourced. This is enabled by
including a `# !source mlflow::mlflow_run` comment at the top of the R
script as
follows:

<img src="tools/readme/mlflow-source-rstudio.png" class="screenshot" width=520 />

## Contributing

In order to contribute, follow the [MLflow contribution
guidelines](../../CONTRIBUTING.rst). After performing python changes,
you can make them available in R by running:

``` r
reticulate::conda_install("r-mlflow", "../../.", pip = TRUE)
```
