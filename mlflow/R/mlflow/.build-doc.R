unlink("man", recursive = TRUE)
roxygen2::roxygenise()
# remove mlflow-package doc temporarily because no rst doc should be generated for it.
file.remove("man/mlflow-package.Rd")
source("document.R", echo = TRUE)
# roxygenize again to make sure the previously removed mlflow-packge doc is available as R helpfile
roxygen2::roxygenise()
