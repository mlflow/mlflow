parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]

library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())

devtools::check_built(path = package, error_on = "note", args = "--no-tests")
source("testthat.R")
