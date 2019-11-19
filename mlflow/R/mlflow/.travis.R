parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)
install.packages("keras", repos='http://cran.rstudio.com/')
install.packages("roxygen2")
library(keras)
# pinning tensorflow version to 1.14 until test_keras_model.R is fixed
install_keras(method = "conda", tensorflow="1.14.0")
devtools::check_built(path = package, error_on = "note", args = "--no-tests")
source("testthat.R")
