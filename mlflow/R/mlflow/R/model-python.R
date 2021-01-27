create_conda_env <- function(name,
                             path,
                             channels = list("defaults"),
                             conda_deps = list(),
                             pip_deps = list()) {
  conda_deps$pip <- pip_deps
  deps <- list(
    name = name,
    channels = channels,
    dependencies = list(conda_deps)
  )
  write_yaml(deps, path)
}

create_pyfunc_conf <- function(loader_module, code = NULL, data = NULL, env = NULL) {
  res <- list(loader_module = loader_module)
  if (!is.null(code)) {
    res$code <- code
  }
  if (!is.null(data)) {
    res$data <- data
  }
  if (!is.null(env)) {
    res$env <- env
  }
  list(python_function = res)
}
