# Increase the timeout length for `utils::download.file` because the default value (60 seconds)
# could be too short to download large packages such as h2o.
options(timeout=300)
install.packages("https://cran.r-project.org/src/contrib/remotes_2.5.0.tar.gz", repos = NULL, type = "source")
remotes::install_version("pak", "0.9.2")
pak::local_install_dev_deps()
remotes::install_version("devtools", "2.4.6")
remotes::install_version("usethis", "3.2.1")

# Install dependencies for documentation build
# Install Rd2md from source as a temporary fix for the rendering of code examples, until
# a release is published including the fixes in https://github.com/quantsch/Rd2md/issues/1
# Note that this commit is equivalent to commit 6b48255 of Rd2md master
# (https://github.com/quantsch/Rd2md/tree/6b4825579a2df8a22898316d93729384f92a756b)
# with a single extra commit to fix rendering of \link tags between methods in R documentation.
remotes::install_git("https://github.com/smurching/Rd2md", ref = "ac7b22bb7452113ea8b2dcaca083f60041e0d4c3")
remotes::install_version("roxygen2", "7.1.2")
remotes::install_version("rmarkdown", "2.30")
