parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)

# The default timeout value (60 seconds) can be insufficient for `spark_install` to complete
options(timeout=60 * 60)

# Install MLeap runtime and required dependencies
sparklyr::spark_install(version = "2.4.5", verbose = TRUE)
