# Computes path to Python executable from the MLFLOW_PYTHON_BIN environment variable.
get_python_bin <- function() {
  in_env <- Sys.getenv("MLFLOW_PYTHON_BIN")
  if (in_env != "") {
    return(in_env)
  }
  # MLFLOW_PYTHON_EXECUTABLE is an environment variable that's defined in a Databricks notebook
  # environment.
  mlflow_python_executable <- Sys.getenv("MLFLOW_PYTHON_EXECUTABLE")
  print("mlflow_python_executable")
  print(mlflow_python_executable)
  if (mlflow_python_executable != "") {
    stdout <- system(paste(mlflow_python_executable, '-c "import sys; print(sys.executable)"' ),
                    intern = TRUE,
                    ignore.stderr = TRUE)
    print("stdout")
    print(stdout)
    return(paste(stdout, collapse = ""))
  }
  python_bin <- Sys.which("python")
  if (python_bin != "") {
    return(python_bin)
  }
  stop(paste("MLflow not configured, please run `pip install mlflow` or ",
             "set MLFLOW_PYTHON_BIN and MLFLOW_BIN environment variables.", sep = ""))
}

# Returns path to Python executable
python_bin <- function() {
  if (is.null(.globals$python_bin)) {
    python <- get_python_bin()
    .globals$python_bin <- path.expand(python)
  }

  .globals$python_bin
}

# Returns path to MLflow CLI, assumed to be in the same bin/ directory as the
# Python executable
python_mlflow_bin <- function() {
  in_env <- Sys.getenv("MLFLOW_BIN")
  print("in_env")
  print(in_env)
  if (in_env != "") {
    return(in_env)
  }
  mlflow_bin <- Sys.which("mlflow")
  print("mlflow_bin")
  print(mlflow_bin)
  if (mlflow_bin != "") {
    return(mlflow_bin)
  }
  print("mlflow_bin")
  print(mlflow_bin)
  python_bin_dir <- dirname(python_bin())
  print("python_bin_dir")
  print(python_bin_dir)
  if (.Platform$OS.type == "windows") {
    file.path(python_bin_dir, "Scripts", "mlflow")
  } else {
    file.path(python_bin_dir, "mlflow")
  }
}
