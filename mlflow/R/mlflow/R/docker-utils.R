#' Build a MLflow Docker image
#'
#' Build a MLflow Docker image that will an RFunc MLflow model
#'
#' @template roxlate-model-uri
#' @param image_name Name of the Docker image
#' @param port Port for serving the model (default: 8090)
#' @param mlflow_version (Optional) Ignored if `mlflow_home` is specified,
#'   otherwise build the image with the  version of MLflow specified by
#'   `mlflow_version` (default: `packageVersion("mlflow")`)
#' @param custom_setup_steps_hook (Optional) single-argument function accepting
#'   the dockerfile context directory as input and returning additional
#'   Dockerfile commands to run during the image build step as output.
#'
#' @examples
#' \dontrun{
#'
#' library(mlflow)
#' library(carrier)
#'
#' # build a model and save it locally
#' model <- lm(Sepal.Width ~ Sepal.Length + Petal.Width, iris)
#' fn <- crate(~ stats::predict(model, .x), model = model)
#'
#' mlflow_save_model(fn, path = "/tmp/model")
#'
#' # now build a self-contained Docker image that can serve that model
#'
#' build_docker_image(
#'   image_name = "mlflow_hello_world",
#'   model_uri = "file:///tmp/model",
#'   port = 8090,
#'   mlflow_version = "1.12.1"
#' )
#'
#' # Now one can, for example, run `docker run -p 9123:8090 mlflow_hello_world`
#' # to launch the docker image above with port 8090 of the container binding to
#' # port 9123 of the host
#'
#'}
#'
#' @export
build_docker_image <- function(image_name,
                               model_uri,
                               port = 8090,
                               mlflow_version = utils::packageVersion("mlflow"),
                               custom_setup_steps_hook = NULL) {
  src_model_path <- mlflow_download_artifacts_from_uri(model_uri)
  dst_model_path <- "/opt/ml/model"
  serve_model_impl <- glue::glue(
    paste0(
      "Sys.setenv(MLFLOW_PYTHON_BIN = '/miniconda/bin/mlflow');",
      "mlflow::mlflow_rfunc_serve(model_uri = 'file://{model_path}', host = '0.0.0.0', port = {port}, daemonized = FALSE)"
    ),
    model_path = dst_model_path,
    port = port
  )
  model_setup_steps_hook <- function(cwd) {
    model_dir <- "model"
    fs::dir_copy(src_model_path, fs::path(cwd, model_dir))
    glue::glue(
      "
        COPY {model_dir} {dst_model_path}
        {additional_steps}
      ",
      model_dir = model_dir,
      dst_model_path = dst_model_path,
      additional_steps = (
        if (is.null(custom_setup_steps_hook)) {
          ""
        } else {
          custom_setup_steps_hook(cwd)
        }
      )
    )
  }

  .build_docker_image(
    image_name = image_name,
    custom_setup_steps_hook = model_setup_steps_hook,
    mlflow_version = mlflow_version,
    entry_point = sprintf(
      "ENTRYPOINT [ \"Rscript\", \"-e\", \"%s\" ]",
      serve_model_impl
    )
  )
}

#' Build a MLflow Docker image
#'
#' Utility function for building a Docker image for serving a MLflow model
#'
#' @param image_name Name of the image
#' @param entry_point String containing ENTRYPOINT directive for the docker
#'   image
#' @param mlflow_home (Optional) Path to local copy of MLflow. If unspecified,
#'   then the image will be built with the version of MLflow specified by
#'   `mlflow_version`.
#' @param mlflow_version (Optional) Ignored if `mlflow_home` is specified,
#'   otherwise build the image with the  version of MLflow specified by
#'   `mlflow_version` (default: `packageVersion("mlflow")`)
#' @param custom_setup_steps_hook (Optional) single-argument function accepting
#'   the dockerfile context directory as input and returning additional
#'   Dockerfile commands to run during the image build step as output.
#'
#' @keywords internal
.build_docker_image <- function(image_name,
                                entry_point,
                                mlflow_home = NULL,
                                mlflow_version = utils::packageVersion("mlflow"),
                                custom_setup_steps_hook = NULL) {
  if (!is.null(mlflow_home)) {
    mlflow_home <- normalizePath(mlflow_home)
  }

  tmp <- tempfile("docker_image_")
  fs::dir_create(tmp)
  on.exit(fs::dir_delete(tmp))

  install_mlflow <- mlflow_docker_installation_steps(tmp, mlflow_home, mlflow_version)
  custom_setup_steps <- (
    if (is.null(custom_setup_steps_hook)) {
      ""
    } else {
      custom_setup_steps_hook(tmp)
    })
  dockerfile <- "Dockerfile"
  dockerfile_template <- read_template_file("DOCKERFILE_TEMPLATE")

  writeLines(
    glue::glue(
      dockerfile_template,
      install_mlflow = install_mlflow,
      custom_setup_steps = custom_setup_steps,
      entry_point = entry_point
    ),
    con = fs::path(tmp, dockerfile)
  )
  message("Building docker image with name ", image_name)
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(tmp)
  system2("docker", c("build", "-t", image_name, "-f", dockerfile, "."))
}

mlflow_docker_installation_steps <- function(dockerfile_context_dir,
                                             mlflow_home,
                                             mlflow_version) {
  if (!is.null(mlflow_home)) {
    mlflow_dir <- copy_project(src = mlflow_home, dst = dockerfile_context_dir)
    template <- read_template_file("COPY_MLFLOW_DIR_TEMPLATE")

    glue::glue(template, mlflow_dir = mlflow_dir)
  } else {
    template <- read_template_file("INSTALL_MLFLOW_TEMPLATE")

    glue::glue(template, version = mlflow_version)
  }
}

#' Copy MLflow project directory for Docker deployment
#'
#' Utility function for copying MLflow project directory for Docker deployment,
#' which will copy everything from the source directory except for paths
#' matching patterns defined in the .dockerignore file.
#' It assumes MLflow is accessible as a local directory.
#'
#' @param src source directory
#' @param dst destination directory
#'
#' @keywords internal
copy_project <- function(src, dst) {
  docker_ignore_file <- fs::path(src, ".dockerignore")
  paths_to_ignore <- (
    if (fs::file_exists(docker_ignore_file)) {
      Sys.glob(fs::path(src, readLines(docker_ignore_file)))
    } else {
      NULL
    })

  mlflow_dir <- "mlflow-project"
  dst <- fs::path(dst, mlflow_dir)
  setup_py_path <- file.path(src, "setup.py")
  if (!fs::is_file(setup_py_path)) {
    stop("file not found ", normalizePath(setup_py_path))
  }
  fs::dir_walk(
    path = src,
    fun = function(path) {
      if (!path %in% paths_to_ignore) {
        src_file <- fs::path_file(path)
        src_dir <- fs::path_dir(path)
        dst_dir <- fs::path(dst, fs::path_rel(src_dir, start = src))
        dst_file <- fs::path(dst_dir, src_file)
        if (fs::is_file(path)) {
          fs::dir_create(dst_dir, recurse = TRUE)
          fs::file_copy(path, dst_file)
        } else {
          fs::dir_create(dst_file, recurse = TRUE)
        }
      }
    },
    all = TRUE
  )

  mlflow_dir
}

read_template_file <- function(template) {
  paste0(
    readLines(system.file(template, package = "mlflow")),
    collapse = "\n"
  )
}
