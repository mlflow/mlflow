# Install Rd2md from source as a temporary fix for the rendering of code examples, until
# a release is published including the fixes in https://github.com/quantsch/Rd2md/issues/1
# Note that this commit is equivalent to commit 6b48255 of Rd2md master
# (https://github.com/quantsch/Rd2md/tree/6b4825579a2df8a22898316d93729384f92a756b)
# with a single extra commit to fix rendering of \link tags between methods in R documentation.
devtools::install_git("https://github.com/smurching/Rd2md", ref = "mlflow-patches")
install.packages("rmarkdown", repos = "https://cloud.r-project.org")
unlink("man", recursive = TRUE)
roxygen2::roxygenise()
# remove mlflow-package doc temporarily because no rst doc should be generated for it.
file.remove("man/mlflow-package.Rd")
source("document.R", echo = TRUE)
# roxygenize again to make sure the previously removed mlflow-packge doc is available as R helpfile
roxygen2::roxygenise()
