parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]

library(mlflow)
library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())
# TODO(harupy): Add `error_on = "note"` once the issue below has been fixed:
# https://stackoverflow.com/questions/63613301/r-cmd-check-note-unable-to-verify-current-time/63616156#63616156
devtools::check_built(path = package, args = "--no-tests")
source("tests/testthat.R", chdir = TRUE)
