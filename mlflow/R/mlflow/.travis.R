parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)
install.packages("keras", repos='http://cran.rstudio.com/')
library(keras)
install_keras()
devtools::check(error_on = "warning")
source("tests/testthat.R")
