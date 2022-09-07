#' @include model.R
#' @include model-python.R
NULL

create_default_conda_env_if_absent <- function(
					       model_path,
					       conda_env,
					       default_conda_deps = list(),
					       default_pip_deps = list()) {
  if (!is.null(conda_env)) {
    dst <- file.path(model_path, basename(conda_env))
    if (conda_env != dst) {
      file.copy(from = conda_env, to = dst)
    }

    basename(conda_env)
  } else { # create default conda environment
    conda_env_file_name <- "conda_env.yaml"
    create_conda_env(
      name = "conda_env",
      path = file.path(model_path, conda_env_file_name),
      conda_deps = default_conda_deps,
      pip_deps = default_pip_deps
    )

    conda_env_file_name
  }
}

create_python_env <- function(model_path,
                              dependencies,
                              build_dependencies = list("pip", "setuptools", "wheel")) {
  python_env_file_name <- "python_env.yaml"
  deps <- list(build_dependencies = build_dependencies, dependencies = dependencies)
  write_yaml(deps, file.path(model_path, python_env_file_name))
  python_env_file_name
}

assert_pkg_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("'%s' package must be installed!", pkg))
  }
}

remove_patch_version <- function(version) {
  gsub("([^.]*\\.[^.]*)(\\..*)", "\\1", version)
}
