parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)

mlflow:::mlflow_maybe_create_conda_env(python_version = "3.6")
library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())
# pinning tensorflow version to 1.14 until test_keras_model.R is fixed
keras::install_keras(method = "conda", envname = mlflow:::mlflow_conda_env_name(), tensorflow="1.15.2")
reticulate::conda_install(Sys.getenv("MLFLOW_HOME", "../../../../."), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
reticulate::conda_install("xgboost", envname = mlflow:::mlflow_conda_env_name())
devtools::check_built(path = package, error_on = "note", args = "--no-tests")
