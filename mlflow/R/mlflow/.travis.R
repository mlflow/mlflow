parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)
install.packages("keras", repos='http://cran.rstudio.com/')
install.packages("roxygen2")
library(keras)
install_keras()
devtools::check(error_on = "warning", args = "--no-tests")
source("testthat.R")
