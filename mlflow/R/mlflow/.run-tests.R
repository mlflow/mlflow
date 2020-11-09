parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]

library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())

devtools::check_built(path = package, args = "--no-tests", error_on = "note")
source("testthat.R")
