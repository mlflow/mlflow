# Install MLflow for R
files <- dir(".", full.names = TRUE)
package <- files[grepl("mlflow_.+\\.tar\\.gz$", files)]
install.packages(package)

mlflow:::mlflow_maybe_create_conda_env(python_version = "3.7")
# Install python dependencies
reticulate::conda_install(Sys.getenv("MLFLOW_HOME", "../../.."), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE, pip_options = c("-e"))
# pinning tensorflow version to 1.14 until test_keras_model.R is fixed
keras::install_keras(method = "conda", envname = mlflow:::mlflow_conda_env_name(), tensorflow="1.15.2")
# pinning h5py < 3.0.0 to avoid this issue:  https://github.com/tensorflow/tensorflow/issues/44467
# TODO: unpin after we use tensorflow >= 2.4
reticulate::conda_install(c("'h5py<3.0.0'", "protobuf<4.0.0"), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
reticulate::conda_install("xgboost", envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
reticulate::conda_install(paste0("h2o==", packageVersion("h2o")), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
