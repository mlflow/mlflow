# Increase the timeout length for `utils::download.file` because the default value (60 seconds)
# could be too short to download large packages such as h2o.
options(timeout=300)
install.packages("devtools", dependencies = TRUE)
devtools::install_dev_deps(dependencies = TRUE)
