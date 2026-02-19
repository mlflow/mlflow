# Increase the timeout length for `utils::download.file` because the default value (60 seconds)
# could be too short to download large packages such as h2o.
options(timeout=300)
install.packages("devtools", dependencies = TRUE)
devtools::install_version("usethis", "3.2.1")
devtools::install_dev_deps(dependencies = TRUE)

# Install dependencies for documentation build
# Install Rd2md from source as a temporary fix for the rendering of code examples, until
# a release is published including the fixes in https://github.com/quantsch/Rd2md/issues/1
# Note that this commit is equivalent to commit 6b48255 of Rd2md master
# (https://github.com/quantsch/Rd2md/tree/6b4825579a2df8a22898316d93729384f92a756b)
# with a single extra commit to fix rendering of \link tags between methods in R documentation.
devtools::install_git("https://github.com/smurching/Rd2md", ref = "mlflow-patches")
devtools::install_version("roxygen2", "7.1.2")
# The latest version of git2r (0.35.0) doesn't work with the rocker/r-ver:4.2.1 docker image
devtools::install_version("git2r", "0.33.0")
install.packages("rmarkdown", repos = "https://cloud.r-project.org")
