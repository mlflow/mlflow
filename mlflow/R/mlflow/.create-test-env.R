parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]
install.packages(package)

mlflow:::mlflow_maybe_create_conda_env(python_version = "3.6")
library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())
# pinning tensorflow version to 1.14 until test_keras_model.R is fixed
keras::install_keras(method = "conda", envname = mlflow:::mlflow_conda_env_name(), tensorflow="1.15.2")
# pinning h5py < 3.0.0 to avoid this issue:  https://github.com/tensorflow/tensorflow/issues/44467
# TODO: unpin after we use tensorflow >= 2.4
reticulate::conda_install("'h5py<3.0.0'", envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
reticulate::conda_install(Sys.getenv("MLFLOW_HOME", "../../../../."), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
reticulate::conda_install("xgboost", envname = mlflow:::mlflow_conda_env_name())
# Pin h2o to prevent version-mismatch between python and R
reticulate::conda_install("h2o==3.30.1.3", envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
