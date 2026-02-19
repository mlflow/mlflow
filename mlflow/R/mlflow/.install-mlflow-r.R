# Install MLflow for R
files <- dir(".", full.names = TRUE)
package <- files[grepl("mlflow_.+\\.tar\\.gz$", files)]
install.packages(package)
